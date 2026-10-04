# Spatial Lipidomics of the Mouse Brain

My working repository for learning spatial metabolomics, specifically MALDI mass spectrometry imaging (MALDI-MSI) of brain lipids.

In this repo I'm working through the **CAJAL Neuromics spatial metabolomics tutorial** from the La Manno lab at EPFL:
[lamanno-epfl/lipidomics_tutorial_cajalcourse](https://github.com/lamanno-epfl/lipidomics_tutorial_cajalcourse)

The tutorial follows the [Lipid Brain Atlas](https://www.biorxiv.org/content/10.1101/2025.10.13.682018v1) (Fusar Bassini et al., 2025) and compares a coronal section from a control female mouse brain with one from a pregnant female. The question at the end: **which brain lipids change during pregnancy, where, and which genes might explain it?**

All credit for the tutorial design, the `cajal_lipidomics` helper package, and the original notebooks goes to the course authors. This repo holds my own runs, notes, figures, and whatever extra analysis I end up doing.

---

## What I'm learning

- Reading MALDI-MSI data and the formats it comes in (METASPACE, AnnData, parquet, zarr)
- Annotating mass peaks to lipid names with ppm matching against an LC-MS reference
- Normalizing across sections with [uMAIA](https://github.com/lamanno-epfl/uMAIA)
- Registering pixels to the Allen Mouse Brain Common Coordinate Framework (CCFv3)
- Feature selection (Moran's I), NMF embedding, t-SNE, and Harmony integration
- Clustering pixels into lipid territories ("lipizones") with Leiden and [EUCLID](https://github.com/lamanno-epfl/EUCLID), and transferring labels from control to pregnant
- Differential lipid testing (Wilcoxon rank-sum + Benjamini-Hochberg) and composite scores like myelination
- Linking lipid changes to gene expression using Allen MERFISH, XGBoost, SHAP, and gene ontology

## Repository structure

```
spatial-lipidomics-project/
├── README.md
├── pyproject.toml           makes src/lipid_project installable (pip install -e .)
├── .gitignore               keeps data, outputs and cloned code out of git
├── notebooks/               my notebooks, one per session (01 to 09)
├── notes/                   my written notes (tutorial_overview.md is the big one)
├── src/lipid_project/
│   └── paths.py             one place that knows where the tutorial, its data, and my outputs live
├── scripts/
│   ├── setup.sh             one-time setup: clones, conda envs, kernels, data download
│   └── check_setup.py       sanity check for an env
├── figures/                 plots and figure panels I want to keep
├── results/                 small summary tables (CSV)
├── data/                    NOT committed
│   └── derived/             my pipeline outputs: 01_raw.h5ad → 02_annotated → 03_normalized → 05_embedded → 06_clustered
└── external/                NOT committed (created by setup.sh)
    ├── lipidomics_tutorial_cajalcourse/   the tutorial: cajal_lipidomics package + provided data
    ├── uMAIA/
    └── EUCLID/
```

## How this repo connects to the tutorial

I don't copy the tutorial's code or notebooks in here. Instead:

- `scripts/setup.sh` clones the tutorial (plus uMAIA and EUCLID) into `external/`, which git ignores.
- The tutorial's helper package `cajal_lipidomics` and my own small package `lipid_project` get installed into the conda envs, so any notebook can `import` both.
- My notebooks never hard-code paths. They use `lipid_project.paths`:

```python
from lipid_project import paths

paths.summary()                                              # what's where, and what's built
reg = paths.tutorial_data("provided", "registration_ccf.parquet")   # tutorial's provided inputs
adata = ad.read_h5ad(paths.stage("normalized"))              # my own outputs in data/derived/
paths.savefig(fig, "myelination_map")                        # figures/ as PDF + PNG
```

If I ever keep the tutorial clone somewhere else, I just set `export LIPID_TUTORIAL_DIR=/path/to/it` before opening Jupyter.

## Setup

Requirements: [Miniforge](https://github.com/conda-forge/miniforge) (mamba/conda) and git. Mac, Linux, or Windows through **WSL2** (notebook 03 needs JAX, which has no native Windows build).

```bash
git clone https://github.com/n3harao/spatial-lipidomics-project.git
cd spatial-lipidomics-project
bash scripts/setup.sh            # or: bash scripts/setup.sh --no-data  to skip the ~1.3 GB download for now
```

That creates two conda environments from the tutorial's own environment files:

| env / kernel | used for |
|---|---|
| `cajal-lipidomics` | every notebook except 03 |
| `cajal-umaia` | notebook 03 only (uMAIA needs an older JAX stack) |

Check an env any time with:

```bash
mamba run -n cajal-lipidomics python scripts/check_setup.py
```

In VS Code, always double check the kernel (top right). A `ModuleNotFoundError` almost always means the wrong kernel.

## Data

The data is too big for GitHub, so none of it is committed.

1. The raw MALDI-MSI sections come from the public Lipid Brain Atlas project on [METASPACE](https://metaspace2020.org/project/mlba-2025?tab=datasets), pulled inside notebook 01.
2. The provided inputs (registration, references, masks, MERFISH, gene ontology) come from the tutorial's Zenodo data bundle, downloaded by `setup.sh` into `external/lipidomics_tutorial_cajalcourse/data/`.
3. Everything I build lands in `data/derived/`. The notebooks run in order, each one loading the previous stage.

## References

- Fusar Bassini et al., *The lipidomic architecture of the mouse brain*, bioRxiv 2025. [link](https://www.biorxiv.org/content/10.1101/2025.10.13.682018v1)
- uMAIA, *Nature Methods* (2025)
- Lipid Brain Atlas explorer: <https://lbae-v2.epfl.ch/>
- Allen Mouse Brain Common Coordinate Framework (CCFv3)

## About me

I'm Neha, a computational biologist. More about me at [n3harao.github.io/Neha-Rao-Portfolio](https://n3harao.github.io/Neha-Rao-Portfolio).
