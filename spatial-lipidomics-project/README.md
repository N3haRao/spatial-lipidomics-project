# Spatial Lipidomics of the Mouse Brain

My working repository for learning spatial metabolomics, specifically MALDI mass spectrometry imaging (MALDI-MSI) of brain lipids.

In this repo I'm working through the **CAJAL Neuromics spatial metabolomics tutorial** from the La Manno lab at EPFL:
👉 [lamanno-epfl/lipidomics_tutorial_cajalcourse](https://github.com/lamanno-epfl/lipidomics_tutorial_cajalcourse)

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

## Progress

| # | Topic | Status |
|---|---|---|
| 00 | Intro: tooling, Python, math & stats | ⬜ |
| 01 | Mass spectra and the raw data | ⬜ |
| 02 | From m/z peaks to lipid names | ⬜ |
| 03 | Normalization with uMAIA | ⬜ |
| 04 | Anatomy: registration and the Allen atlas | ⬜ |
| 05 | Embedding: Moran's I, NMF, t-SNE, Harmony | ⬜ |
| 06 | Clustering and label transfer | ⬜ |
| 07 | Clustering from scratch + pregnancy changes | ⬜ |
| 08 | Which genes explain the lipid changes | ⬜ |
| 09 | My own analysis | ⬜ |

(⬜ not started · 🟨 in progress · ✅ done)

## Repository structure

```
spatial-lipidomics-project/
├── README.md          you are here
├── .gitignore         keeps big data files and outputs out of git
├── notebooks/         my notebooks as I work through each session
├── notes/             my written notes on each step (tutorial_overview.md is the big one)
├── src/               any helper code I write myself
├── figures/           plots and figure panels I want to keep
├── results/           small summary tables (CSV) from my analyses
└── data/              local data only, NOT committed (see below)
```

## Data

The data is too big for GitHub, so nothing in `data/` gets committed. To get it:

1. The raw MALDI-MSI sections come from the public Lipid Brain Atlas project on [METASPACE](https://metaspace2020.org/project/mlba-2025?tab=datasets), pulled inside notebook 01.
2. The supporting files (registration, references, masks, MERFISH, gene ontology) come from the tutorial's Zenodo bundle via its `scripts/fetch_data_bundle.py`.

## Setup

I'm following the tutorial's own [setup guide](https://github.com/lamanno-epfl/lipidomics_tutorial_cajalcourse/blob/main/docs/SETUP.md). Short version:

- Two conda environments: `cajal-lipidomics` for most notebooks, and `cajal-umaia` for notebook 03 only (uMAIA needs JAX).
- Install the tutorial's helper package with `pip install -e .` from the tutorial folder.
- In VS Code, always double check the kernel. Wrong kernel = `ModuleNotFoundError`.

## References

- Fusar Bassini et al., *The lipidomic architecture of the mouse brain*, bioRxiv 2025. [link](https://www.biorxiv.org/content/10.1101/2025.10.13.682018v1)
- uMAIA, *Nature Methods* (2025)
- Lipid Brain Atlas explorer: <https://lbae-v2.epfl.ch/>
- Allen Mouse Brain Common Coordinate Framework (CCFv3)

## About me

I'm Neha, a computational biologist. More about me at [n3harao.github.io/Neha-Rao-Portfolio](https://n3harao.github.io/Neha-Rao-Portfolio).
