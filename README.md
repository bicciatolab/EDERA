This repository pertains to the EDERA algorithm: submitted for consideration to RECOMB 2027.
Below we provide instructions on installation and a quick run example.

#### Conda / mamba — recommended

```bash
git clone https://github.com/bicciatolab/EDERA.git
cd EDERA/package

mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate edera
pip install -e .                        # installs EDERA itself
```

#### pip

```bash
cd EDERA/package

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     
pip install -e .
```

Or let `pyproject.toml` resolve the dependency set with loose bounds:

```bash
pip install -e .                         # core
pip install -e ".[html,checkpoint,dev]"  # + plotly, pyarrow, pytest
```

The modules live directly in `package/`; `pyproject.toml` maps them to the importable `EDERA` package, so `import EDERA` works after an editable install without renaming the tree.<a id="quickstart"></a>

#### Quick run

Fast way to obtain results on a test dataset. In alternative we make an explorative notebook available in `notebooks/` to interactively test and play with the algorithm.

```python
import EDERA

df = EDERA.run_pipeline(
    data_path="data/test_dataset.csv",
    x_name="x", y_name="y", z_name="z", gene_name="gene_id",
    out_path="run_fixture",
    plot_format=["png"],
    diagnostic_bbox=(0, 100, 0, 100),
    dpi=150,
    compute_kwargs=dict(
        x_res=1.0, y_res=1.0, z_res=1.0,   # fixture is already in microns
        knn_neighbors=15, max_edge_length=4.0,
        n_topics_lda=5, stopwords_thr=0.4,
    ),
    cluster_kwargs=dict(
        n_neighbors_hnsw=15,
        leiden_resolution=[0.1, 0.5],
        sample_fraction_transfer_cluster=0.5,
    ),
)
print(df[["x", "y", "final_cluster_res0.1", "final_cluster_res0.5"]].head())
```
