"""
EDERA.console
===============

Human-readable terminal logging for the pipeline entry points.

A single module-level :data:`console` (an instance of :class:`PipelineConsole`)
is shared by :mod:`EDERA.api` and every step module, so that the per-step
progress messages, timings and delivered-output paths are formatted
consistently without threading a logger object through every function
signature.

Design notes
------------
* **Zero new dependencies.** Styling uses raw ANSI escape codes. Colour is
  auto-detected: it is enabled only when ``stdout`` is a TTY and ``NO_COLOR``
  is unset, so redirecting the log to a file yields clean plain text.
* **Quiet switch.** Every printing method is gated by ``console.verbose``. When
  a run is started with ``verbose=False`` nothing is printed, but delivered
  paths are still recorded (harmless, and keeps the code paths symmetric).
* **What it produces.** A top banner, one ``┏━ FUNCTION … ━┓`` section per entry
  point, a timed ``▶ Step n/total`` block per pipeline step (each declaring the
  files it wrote), and a final ``PIPELINE COMPLETE`` panel that lists every
  delivered artifact grouped by kind, with the total wall-clock time.
"""

import os
import sys
import time
import shutil
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# Colour / TTY layer
# ---------------------------------------------------------------------------
def _detect_color():
    """Decide whether ANSI colour should be emitted.

    Priority: ``NO_COLOR`` (any value) forces plain text; ``FORCE_COLOR`` (any
    value) forces colour. Otherwise colour is on for a real TTY *and* for a
    Jupyter/IPython notebook kernel — the latter is not a TTY but renders ANSI
    SGR codes in its output cells, so ``isatty()`` alone would wrongly disable it.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if sys.stdout.isatty():
        return True
    try:  # notebooks render ANSI even though stdout is not a TTY
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell":
            return True
    except Exception:
        pass
    return False


_SUPPORTS_COLOR = _detect_color()

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
MAGENTA = "\x1b[35m"
RED = "\x1b[31m"


def _c(text, *codes):
    """Wrap ``text`` in ANSI ``codes`` when colour is enabled, else return it raw."""
    if not _SUPPORTS_COLOR or not codes:
        return text
    return "".join(codes) + text + RESET


def format_duration(seconds):
    """Format a duration in seconds as ``"12.3 s"`` / ``"1 m 45 s"`` / ``"2 h 03 m"``."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        m, s = divmod(int(round(seconds)), 60)
        return f"{m} m {s:02d} s"
    h, rem = divmod(int(round(seconds)), 3600)
    m = rem // 60
    return f"{h} h {m:02d} m"


# Fixed box width, capped to the terminal so narrow windows do not wrap the rules.
_BOX_W = min(74, max(48, shutil.get_terminal_size((80, 24)).columns - 2))
_INNER = _BOX_W - 4  # content width between "║ " and " ║"

# Shared indent for sub-step lines and progress bars (keeps tqdm aligned with
# info()/deliverable() output).
_INDENT = "        "

# Deliverable kinds, in the order they are listed in the final summary.
_KIND_ORDER = ("table", "plot", "array", "checkpoint")
_KIND_LABEL = {"table": "tables", "plot": "plots", "array": "arrays",
               "checkpoint": "checkpoints"}


# ---------------------------------------------------------------------------
# The shared console
# ---------------------------------------------------------------------------
class PipelineConsole:
    """Stateful, colour-aware progress printer shared across the package.

    Not thread-safe by design: it is only driven from the main process of the
    entry points (joblib worker bodies do not log through it).
    """

    def __init__(self):
        self.verbose = True
        self._active = False          # a run is in progress
        self._deliverables = []       # list of (kind, path)
        self._run_start = None
        self._out_path = None
        self._step = None             # (n, total, title, t0, buffer) of active step

    # -- run lifecycle ----------------------------------------------------
    def begin(self, verbose=True, title="pipeline", out_path=None, meta=None):
        """Start a run (or join one already in progress).

        Returns True if this call *owns* the run (it started it and must end
        it); False if a run was already active — the nested case used when
        ``run_pipeline`` wraps the two entry points, so the banner/summary are
        printed once, not three times.
        """
        if self._active:
            return False
        self.verbose = verbose
        self._active = True
        self._deliverables = []
        self._run_start = time.time()
        self._out_path = out_path
        self._banner(title, out_path, meta or {})
        return True

    def end(self, owns):
        """Finish the run: print the summary and reset state (only if ``owns``)."""
        if not owns:
            return
        self._summary()
        self._active = False
        self._step = None

    # -- function-level sections -----------------------------------------
    def function_start(self, name):
        """Print the opening rule for one entry-point section and start its timer."""
        self._fn_start = time.time()
        self._emit("")
        self._emit(self._rule_open(f" FUNCTION · {name} "))

    def function_end(self, name):
        """Print the closing rule with the section's elapsed time."""
        elapsed = time.time() - getattr(self, "_fn_start", time.time())
        self._emit(self._rule_close(
            f" {name} complete · elapsed {format_duration(elapsed)} "))

    # -- steps ------------------------------------------------------------
    @contextmanager
    def step(self, n, total, title):
        """Context manager for one pipeline step: header, timing, done line.

        Deliverables recorded inside the block are flushed as ``➜`` lines after
        the ``✔ done`` line. On an exception a red failure line is printed and
        the exception re-raised.
        """
        self._emit("")
        self._emit(f"  {_c('▶', BOLD)}  "
                   f"{_c(f'Step {n}/{total}', BOLD)}  {_c(title, CYAN)}")
        t0 = time.time()
        prev, self._step = self._step, [n, total, title, t0, []]
        try:
            yield self
        except BaseException:
            self._emit(f"     {_c('✖', RED, BOLD)}  "
                       f"{_c(f'{n}/{total} FAILED', RED)}  "
                       f"{_c('· ' + format_duration(time.time() - t0), DIM)}")
            self._step = prev
            raise
        else:
            buffer = self._step[4]
            self._emit(f"     {_c('✔', GREEN, BOLD)}  {n}/{total} done  "
                       f"{_c('· ' + format_duration(time.time() - t0), DIM)}")
            for kind, path in buffer:
                self._emit_deliverable(kind, path)
            self._step = prev

    def info(self, msg):
        """Print an indented, dimmed sub-step line."""
        self._emit(f"{_INDENT}{_c(str(msg), DIM)}")

    def warn(self, msg):
        """Print an indented warning line (yellow)."""
        self._emit(f"{_INDENT}{_c('! ' + str(msg), YELLOW)}")

    def pbar(self, iterable=None, *, desc="", **kwargs):
        """Wrap ``tqdm`` so progress bars align with the other sub-step lines.

        Indents the bar by the shared sub-step indent and disables it entirely
        in quiet mode (``verbose=False``). All other tqdm keyword arguments
        (``total``, ``leave`` …) pass straight through.
        """
        from tqdm.auto import tqdm
        kwargs.setdefault("disable", not self.verbose)
        return tqdm(iterable, desc=f"{_INDENT}{desc}" if desc else desc, **kwargs)

    # -- deliverables -----------------------------------------------------
    def deliverable(self, path, kind="table"):
        """Record one or more delivered output paths and (if inside a step) show them.

        ``path`` may be a single path string or an iterable of paths (plot
        helpers return a list of written files). Empty / falsy entries are
        ignored, so a skipped bbox-gated plot (which returns ``[]``) records
        nothing.
        """
        if not path:
            return
        paths = [path] if isinstance(path, (str, bytes, os.PathLike)) else list(path)
        for p in paths:
            if not p:
                continue
            p = os.fspath(p)
            self._deliverables.append((kind, p))
            if self._step is not None:
                self._step[4].append((kind, p))   # flushed at step end
            else:
                self._emit_deliverable(kind, p)

    # -- rendering helpers ------------------------------------------------
    def _emit(self, line):
        if self.verbose:
            print(line)

    def _emit_deliverable(self, kind, path):
        self._emit(f"{_INDENT}{_c('➜', MAGENTA)} "
                   f"{_c(f'{kind:<10}', MAGENTA)} {_c(self._rel(path), DIM)}")

    def _rel(self, path):
        # Prefer a relative path, but only when it stays within the working tree;
        # a rel path that climbs out with "../../.." is noisier than the absolute
        # one, so fall back to the path as given.
        try:
            rel = os.path.relpath(path)
        except ValueError:
            return path
        return path if rel.startswith("..") else rel

    def _rule_open(self, label):
        fill = max(0, _BOX_W - 4 - len(label))
        return _c(f"┏━━{label}{'━' * fill}┓", CYAN, BOLD)

    def _rule_close(self, label):
        fill = max(0, _BOX_W - 4 - len(label))
        return _c(f"┗━━{label}{'━' * fill}┛", CYAN, BOLD)

    def _box_top(self):
        return _c("╔" + "═" * (_BOX_W - 2) + "╗", CYAN, BOLD)

    def _box_bottom(self):
        return _c("╚" + "═" * (_BOX_W - 2) + "╝", CYAN, BOLD)

    def _box_line(self, content, vis=None):
        vis = len(content) if vis is None else vis
        if vis > _INNER and "\x1b" not in content:
            # Truncate over-long plain content so the right border stays aligned
            # (crafted lines that carry ANSI codes are pre-sized and never here).
            content = content[:_INNER - 1] + "…"
            vis = _INNER
        if vis < _INNER:
            content = content + " " * (_INNER - vis)
        bar = _c("║", CYAN, BOLD)
        return f"{bar} {content} {bar}"

    # -- banner / summary -------------------------------------------------
    def _banner(self, title, out_path, meta):
        if not self.verbose:
            return
        from . import __version__
        self._emit("")
        self._emit(self._box_top())
        head = f"EDERA · {title}"
        ver = f"v{__version__}"
        self._emit(self._box_line(
            _c(head, BOLD) + " " * max(1, _INNER - len(head) - len(ver)) + _c(ver, DIM),
            vis=len(head) + max(1, _INNER - len(head) - len(ver)) + len(ver)))
        if meta.get("data"):
            self._emit(self._box_line(f"input   →  {self._rel(meta['data'])}"))
        if out_path is not None:
            self._emit(self._box_line(f"output  →  {self._rel(out_path)}"))
        bits = []
        if "workers" in meta:
            bits.append(f"workers {meta['workers']}")
        if "plots" in meta and meta["plots"]:
            bits.append("plots " + ", ".join(meta["plots"]))
        if "seed" in meta:
            bits.append(f"seed {meta['seed']}")
        if bits:
            self._emit(self._box_line("   ·   ".join(bits)))
        self._emit(self._box_bottom())

    def _summary(self):
        if not self.verbose:
            return
        elapsed = time.time() - (self._run_start or time.time())
        n = len(self._deliverables)
        self._emit("")
        self._emit(self._box_top())
        # No emoji inside the box: dingbats/emoji render 1 or 2 cells depending
        # on the terminal, which would misalign the right border. The green bold
        # text carries the "success" cue instead.
        label = "PIPELINE COMPLETE"
        head = _c(label, GREEN, BOLD)
        tail = f"total {format_duration(elapsed)}"
        pad = max(1, _INNER - len(label) - len(tail))
        self._emit(self._box_line(head + " " * pad + _c(tail, DIM),
                                  vis=len(label) + pad + len(tail)))
        self._emit(self._box_bottom())


# Module-level singleton shared across the package.
console = PipelineConsole()
