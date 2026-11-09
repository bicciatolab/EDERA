"""
EDERA.lda_vb
==============

Online variational Bayes LDA (Hoffman, Blei & Bach 2010) compiled with numba.

A line-for-line port of scikit-learn's ``LatentDirichletAllocation``
(``learning_method="online"``): the same per-document E-step
(``_update_doc_distribution``), the same ``exp(E[log theta])`` fixed point,
convergence test and online M-step. Two differences, both deliberate:

* **Threads, not processes.** Documents are tiny here (~18 tokens), so the
  per-document work is a few microseconds and sklearn/gensim spend it almost
  entirely in interpreter overhead; sklearn's ``n_jobs`` adds per-update
  process dispatch on top. Here the E-step is one compiled ``prange`` loop over
  the mini-batch in shared memory.
* **Reproducible at any thread count.** Each document writes only its own row
  of ``gamma`` and its own slice of a per-token buffer; the sufficient
  statistics are then reduced one topic row per thread, each row summed in
  token order. No floating-point sum ever depends on scheduling, so the fit
  is bit-identical for every ``n_jobs`` (sklearn's is only for a fixed one).
* **Per-document ``gamma`` initialisation from a counter-based RNG.** sklearn
  draws the ``Gamma(100, 0.01)`` initialisation for a whole mini-batch from
  one ``RandomState`` in the calling thread, a serial step that cost more than
  the rest of the update's serial work combined. Here each document draws its
  own from a SplitMix64 stream keyed on ``(random_state, epoch, document)``,
  inside the parallel loop. Same distribution, different values;
  ``init="sklearn"`` restores sklearn's exact draws (used by the parity test).

``digamma`` is implemented locally (numba has no ``scipy.special``); it agrees
with scipy to ~1e-15 relative over the range LDA uses.
"""

import numpy as np
import numba
from numba import njit, prange

_EPS = np.finfo(np.float64).eps


@njit(cache=True, fastmath=False)
def _digamma(x):
    """psi(x) for x > 0: recurrence to x >= 6, then the asymptotic series."""
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    t = f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (
        1.0 / 240.0 + f * (-1.0 / 132.0)))))
    return r + np.log(x) - 0.5 / x + t


@njit(parallel=True, cache=True)
def _exp_dirichlet_expectation_2d(arr):
    """``exp(psi(arr) - psi(arr.sum(1)))`` row-wise, returned **transposed**
    (n_features x n_topics) so the E-step reads one word's topics contiguously.
    """
    K, V = arr.shape
    out = np.empty((V, K))
    for k in prange(K):
        tot = 0.0
        for v in range(V):
            tot += arr[k, v]
        psi_tot = _digamma(tot)
        for v in range(V):
            out[v, k] = np.exp(_digamma(arr[k, v]) - psi_tot)
    return out


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


@njit(cache=True, inline="always")
def _splitmix64(state):
    """One SplitMix64 step: returns ``(new_state, 64-bit output)``."""
    state = state + _GOLDEN
    z = state
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return state, z ^ (z >> np.uint64(31))


@njit(cache=True, inline="always")
def _uniform(state):
    """Uniform in (0, 1) from the top 53 bits; never exactly 0."""
    state, z = _splitmix64(state)
    return state, ((z >> np.uint64(11)) + np.uint64(1)) * (1.0 / 9007199254740993.0)


@njit(cache=True)
def _gamma_init(g, seed, epoch, doc):
    """Fill ``g`` with Gamma(100, 0.01) draws (sklearn's init distribution),
    from a stream keyed on ``(seed, epoch, doc)`` — Marsaglia-Tsang with a
    Box-Muller normal. Independent of which thread runs the document."""
    state = (np.uint64(seed) * _MIX1) ^ (np.uint64(epoch) * _MIX2) ^ np.uint64(doc)
    state, _ = _splitmix64(state)
    d = 100.0 - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    for k in range(g.shape[0]):
        while True:
            state, u1 = _uniform(state)
            state, u2 = _uniform(state)
            x = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            v = 1.0 + c * x
            if v <= 0.0:
                continue
            v = v * v * v
            state, u = _uniform(state)
            if np.log(u) < 0.5 * x * x + d - d * v + d * np.log(v):
                g[k] = 0.01 * d * v
                break


@njit(parallel=True, cache=True)
def _e_step(indptr, indices, data, exp_beta_t, gamma, prior, max_iter, tol,
            phi_tok, seed, epoch, doc0):
    """Variational E-step over one block of documents, in place.

    ``gamma`` (n_docs x K) holds the fitted per-document Dirichlet parameters
    on exit. On entry it is the initialisation if ``seed < 0``; otherwise
    document ``d`` is initialised by :func:`_gamma_init` with key
    ``(seed, epoch, doc0 + d)``. If ``phi_tok`` has rows, the
    per-token sufficient-statistic contributions ``exp_theta_d * cnt / norm``
    are written to ``phi_tok[token]`` (n_tokens x K) for a later ordered
    scatter; pass a 0-row array to skip them (inference).
    """
    n_docs, K = gamma.shape
    want_sstats = phi_tok.shape[0] > 0
    for d in prange(n_docs):
        s = indptr[d]
        n = indptr[d + 1] - s
        g = gamma[d]
        if seed >= 0:
            _gamma_init(g, seed, epoch, doc0 + d)
        et = np.empty(K)
        norm = np.empty(n)
        # exp(E[log theta]) of the initial gamma (no prior added, as sklearn).
        tot = 0.0
        for k in range(K):
            tot += g[k]
        psi_tot = _digamma(tot)
        for k in range(K):
            et[k] = np.exp(_digamma(g[k]) - psi_tot)

        for _ in range(max_iter):
            for j in range(n):
                w = indices[s + j]
                acc = 0.0
                for k in range(K):
                    acc += et[k] * exp_beta_t[w, k]
                norm[j] = data[s + j] / (acc + _EPS)
            change = 0.0
            tot = 0.0
            for k in range(K):
                acc = 0.0
                for j in range(n):
                    acc += norm[j] * exp_beta_t[indices[s + j], k]
                new = et[k] * acc + prior
                change += abs(new - g[k])
                g[k] = new
                tot += new
            psi_tot = _digamma(tot)
            for k in range(K):
                et[k] = np.exp(_digamma(g[k]) - psi_tot)
            if change / K < tol:
                break

        if want_sstats:
            for j in range(n):
                w = indices[s + j]
                acc = 0.0
                for k in range(K):
                    acc += et[k] * exp_beta_t[w, k]
                c = data[s + j] / (acc + _EPS)
                for k in range(K):
                    phi_tok[s + j, k] = et[k] * c


@njit(parallel=True, cache=True)
def _scatter_sstats(indices, phi_tok, sstats):
    """``sstats[:, w] += phi`` with one topic row per thread, each summed in
    token (= document) order — the fixed order that makes the fit independent
    of the thread count."""
    for k in prange(sstats.shape[0]):
        row = sstats[k]
        for t in range(indices.shape[0]):
            row[indices[t]] += phi_tok[t, k]


class OnlineLDA:
    """Online VB LDA with sklearn-compatible attributes and semantics.

    Parameters mirror ``sklearn.decomposition.LatentDirichletAllocation``
    (online mode). After :meth:`fit`, ``components_`` is the variational
    topic-word parameter lambda (n_topics x n_features).
    """

    def __init__(self, n_components=10, doc_topic_prior=None,
                 topic_word_prior=None, learning_decay=0.7,
                 learning_offset=10.0, batch_size=128, max_iter=10,
                 max_doc_update_iter=100, mean_change_tol=1e-3, n_jobs=1,
                 random_state=None, init="counter", schedule="sklearn"):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.max_doc_update_iter = max_doc_update_iter
        self.mean_change_tol = mean_change_tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        if init not in ("counter", "sklearn"):
            raise ValueError(f"init must be 'counter' or 'sklearn'; got {init!r}")
        self.init = init
        if schedule not in ("sklearn", "gensim"):
            raise ValueError(
                f"schedule must be 'sklearn' or 'gensim'; got {schedule!r}")
        self.schedule = schedule

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _as_csr(X):
        X = X.tocsr()
        return (X.indptr.astype(np.int64), X.indices.astype(np.int64),
                X.data.astype(np.float64))

    def _threads(self):
        n = int(self.n_jobs) if self.n_jobs and self.n_jobs > 0 \
            else numba.config.NUMBA_NUM_THREADS
        return max(1, min(n, numba.config.NUMBA_NUM_THREADS))

    def _run(self, fn, *args):
        prev = numba.get_num_threads()
        numba.set_num_threads(self._threads())
        try:
            return fn(*args)
        finally:
            numba.set_num_threads(prev)

    # -- API ---------------------------------------------------------------
    def fit(self, X):
        n_docs, n_features = X.shape
        K = self.n_components
        self.doc_topic_prior_ = (1.0 / K if self.doc_topic_prior is None
                                 else float(self.doc_topic_prior))
        self.topic_word_prior_ = (1.0 / K if self.topic_word_prior is None
                                  else float(self.topic_word_prior))
        rng = np.random.RandomState(self.random_state)
        # Same lambda initialisation draws as sklearn.
        self.components_ = rng.gamma(100.0, 0.01, (K, n_features))
        # Key for the per-document gamma streams (init="counter"): derived
        # from the same RandomState, so random_state=None still varies.
        seed = -1 if self.init == "sklearn" else int(rng.randint(0, 2**31 - 1))
        exp_beta_t = _exp_dirichlet_expectation_2d(self.components_)
        self.n_batch_iter_ = 1
        indptr, indices, data = self._as_csr(X)

        def _fit():
            nonlocal exp_beta_t
            for epoch in range(self.max_iter):
                for start in range(0, n_docs, self.batch_size):
                    stop = min(start + self.batch_size, n_docs)
                    t0, t1 = indptr[start], indptr[stop]
                    b_indptr = indptr[start:stop + 1] - t0
                    b_idx = indices[t0:t1]
                    if seed < 0:
                        gamma = rng.gamma(100.0, 0.01, (stop - start, K))
                    else:
                        gamma = np.empty((stop - start, K))
                    phi_tok = np.empty((t1 - t0, K))
                    _e_step(b_indptr, b_idx, data[t0:t1], exp_beta_t, gamma,
                            self.doc_topic_prior_, self.max_doc_update_iter,
                            self.mean_change_tol, phi_tok, seed, epoch, start)
                    sstats = np.zeros((K, n_features))
                    _scatter_sstats(b_idx, phi_tok, sstats)
                    sstats *= exp_beta_t.T
                    # Online M-step (sklearn _em_step, batch_update=False).
                    # Step size. sklearn: (offset + t)^-decay, t from 1.
                    # gensim: (offset + epoch + t)^-decay, t from 0, so the
                    # first update fully replaces the random lambda and later
                    # epochs decay faster.
                    if self.schedule == "sklearn":
                        base = self.learning_offset + self.n_batch_iter_
                    else:
                        base = (self.learning_offset + epoch
                                + self.n_batch_iter_ - 1)
                    rho = base ** (-self.learning_decay)
                    ratio = float(n_docs) / (stop - start)
                    self.components_ *= 1.0 - rho
                    self.components_ += rho * (self.topic_word_prior_
                                               + ratio * sstats)
                    exp_beta_t = _exp_dirichlet_expectation_2d(self.components_)
                    self.n_batch_iter_ += 1

        self._run(_fit)
        self._exp_beta_t = exp_beta_t
        return self

    def transform(self, X, block=200_000):
        """Normalised per-document topic mixtures (constant ``gamma`` init,
        as sklearn's ``transform``: deterministic and thread-count-free).
        Processed in row blocks so the working set stays bounded."""
        indptr, indices, data = self._as_csr(X)
        n_docs, K = X.shape[0], self.n_components
        out = np.empty((n_docs, K), dtype=np.float64)
        no_phi = np.empty((0, K))
        for start in range(0, n_docs, block):
            stop = min(start + block, n_docs)
            t0 = indptr[start]
            gamma = np.ones((stop - start, K))
            self._run(_e_step, indptr[start:stop + 1] - t0,
                      indices[t0:indptr[stop]], data[t0:indptr[stop]],
                      self._exp_beta_t, gamma, self.doc_topic_prior_,
                      self.max_doc_update_iter, self.mean_change_tol, no_phi,
                      -1, 0, 0)
            out[start:stop] = gamma / gamma.sum(axis=1, keepdims=True)
        return out
