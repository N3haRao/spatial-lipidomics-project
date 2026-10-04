# Spatial Lipidomics Tutorial (CAJAL course) · My Notes

**Repo:** https://github.com/lamanno-epfl/lipidomics_tutorial_cajalcourse
**Course:** CAJAL Neuromics summer school, Bordeaux, July 2026 · instructor Luca Fusar Bassini (EPFL, La Manno lab)
**Paper it's based on:** Fusar Bassini et al., *The lipidomic architecture of the mouse brain*, bioRxiv 2025 (the Lipid Brain Atlas, aka LBA)
**Explore the atlas:** https://lbae-v2.epfl.ch/

---

The whole tutorial is one long pipeline. I take MALDI mass spec imaging data from two mouse brain sections (one control female, one pregnant female, same coronal plane around AP 6.5), and I walk it all the way from raw m/z peaks to "which genes might explain the lipid changes in pregnancy."

The headline result I'm supposed to reproduce: **myelin sphingolipids (HexCer, SM, Cer) go up in pregnant white matter**, and the gene program that best explains where lipids change is a **myelination / oligodendrocyte** program.

The vibe of the course: nothing is a black box. Every step gets "unrolled" in plain numpy first, then I call the helper from `cajal_lipidomics` and check it matches. The repeated instruction is literally "open the `.py` file and read the function before you run it."

---

## The pipeline at a glance

Each notebook saves an `.h5ad` that the next one loads. It's a chain, so I have to run them in order.

| # | Notebook | What I do | Output file |
|---|---|---|---|
| 00 | intro (3 notebooks) | terminal, git, Python, linear algebra, stats | none |
| 01 | mass spectra & data | pull 2 sections from METASPACE, build pixels x ions matrix, attach Allen anatomy | `01_raw.h5ad` |
| 02 | annotation | turn m/z peaks into lipid names (ppm matching vs LC-MS) | `02_annotated.h5ad` |
| 03 | normalization (uMAIA) | remove section-to-section batch effect | `03_normalized.h5ad` |
| 04 | anatomy & Allen atlas | understand registration, grey vs white matter, region x lipid matrix | (no new file, uses 03) |
| 05 | embedding | Moran's I feature selection → NMF → t-SNE → Harmony | `05_embedded.h5ad` |
| 06 | clustering & label transfer | Leiden lipizones on control, kNN transfer to pregnant, run EUCLID | `06_clustered.h5ad` |
| 07 | clustering from scratch + differential | build a divisive splitter, Wilcoxon + BH, pregnancy changes, composite scores | none |
| 08 | multimodal (genes) | join with Allen MERFISH, XGBoost + SHAP + gene ontology | none |
| 09 | my own analysis | open-ended project, AI-assisted | whatever I make |

The data object the whole time is one AnnData with **189,011 pixels x 104 ions** (control `naive` = 88,753 pixels, SectionID 75; `pregnant` = 100,258 pixels, SectionID 110).

---

## Setup notes (so I don't get stuck)

- **Two conda envs**, and this matters:
  - `cajal-lipidomics`: the main one, used by basically every notebook
  - `cajal-umaia`: **only for notebook 03**, because uMAIA needs JAX + `numpy<2`
- Use Miniforge / mamba, not full Anaconda.
- Windows → has to be **WSL2** (no Windows build of `jaxlib`, so NB03 won't run natively).
- After creating each env: `pip install -e .` so `cajal_lipidomics` imports, then register the kernel with `ipykernel`.
- Data bundle (~1 GB on Zenodo, registration + references + masks + MERFISH + GO files): `python scripts/fetch_data_bundle.py`
- Raw MALDI I pull myself from METASPACE inside NB01.
- EUCLID isn't installed up front, NB06 git-clones it on the fly.
- **#1 bug to expect:** `ModuleNotFoundError` almost always means I'm on the wrong kernel (e.g. `base`). Fix: re-pick the kernel in VS Code. Sanity check: `import sys; print(sys.executable)` should end in `envs/cajal-lipidomics/bin/python`.

### The helper package `cajal_lipidomics` (imported as `cl`)

Modules I'll keep bumping into: `annotation`, `analysis`, `embedding`, `plotting`, `multimodal`, `ml`, `data`, `style`. Quick map of the ones that matter most:

- `annotation`: `ppm_error`, `match_lcms`, `plot_ppm_match`, `ADDUCTS`
- `analysis`: `min01_per_lipid`, `morans_i`, `differential_lipids`, `marker_lipids`, `myelination_score`, `membrane_remodeling_score`
- `embedding`: `seeded_nmf`, `apply_nmf`, `harmonize`, `leiden_clusters`, `knn_transfer`
- `multimodal`: `region_change_matrix`, `join_genes`, `gene_programs`, `predict_changes`, `match_pixels_to_cells`, `reciprocal_enrichment`, `top_genes_for_program`
- `plotting`: `spatial_lipid`, `spatial_categorical`, `sorted_lipid_heatmap`, `volcano`, `allen_contours`, `rgb_overlay`, etc.

---

## 00 · Intro (self-guided prep)

Three notebooks, meant to be done before the course. I can skim these since most is review for me, but the stats one is worth a look because it sets up stuff used later.

**00_tooling:** terminal basics (`ls`, `cd`, `mkdir`, `--help`), git's four verbs, why environments exist, Jupyter run order + kernels, and how to use Claude Code responsibly ("read what it writes").

**01_python_for_data:** variables/lists/dicts/loops/functions, numpy (shape, slicing, vectorized math, `axis`, boolean masks), pandas (`value_counts`, `groupby`), matplotlib (fig vs axes), and a first look at AnnData.

**02_concepts (the useful one):**
- Linear algebra: vectors as arrows, dot product → length + angle, **cosine similarity** (direction not magnitude, so a pixel `[8,2]` and `[4,1]` have the same lipid *composition*), matrices as both data tables and transformations, rank (true dimensionality), projection (the core of least squares and dim reduction), eigenvectors of a covariance matrix = PCA directions.
- Stats: normal distribution + CLT, SEM = σ/√n, Pearson r (only sees linear stuff, a U-shape gives r ≈ 0), least squares + R², `curve_fit` with a Michaelis-Menten example, t-tests, confidence intervals, and a teaser for **rank-based tests** (Wilcoxon/Mann-Whitney) which come back in NB07.

---

## 01 · Mass spectra and the raw data

**Goal:** build the raw data matrix myself from public data.

### The physics, in my words
- Thin (~10 µm) frozen brain slice, sprayed with a **matrix** (DHB) that co-crystallizes with the tissue molecules.
- Laser hits a spot → matrix absorbs energy and blasts off, taking tissue molecules with it into the gas phase (that's the "desorption"). They pick up charge → ions.
- Mass spec sorts ions by **m/z** and counts them → one **mass spectrum** per spot.
- Raster the laser across the slice every **25 µm** → each grid point is a **pixel**.
- Instrument details for both sections: Orbitrap, positive ion mode, resolving power R = 240,000 at m/z 200, 25 µm pixels, m/z 400 to 1600, DHB matrix.

> **A pixel is not a cell.** 25 µm is bigger than a neuron body. Each pixel is a mix of cell bodies, axons, dendrites, glia, ECM. So a lipid map shows bulk chemistry of a tissue patch.

### Brain lipids 101
- Brain is ~50% lipid by dry weight. Lipids make membranes, synaptic vesicles, and **myelin** (why white matter is white).
- Classes I'll see a lot:
  - Glycerophospholipids: **PC** (most abundant), **PE**, **PS**, **PI**, PA, PG, and lyso forms **LPC / LPE** (single chain)
  - Sphingolipids (myelin-heavy): **SM**, **Cer**, **HexCer** (big myelin marker, comes up constantly)
- Reading a name: `PC 38:6` = class PC, 38 total carbons across both chains, 6 total double bonds. `HexCer 42:2;O2` adds two backbone oxygens.
- These are **sum compositions**. MALDI can't tell `18:0/20:6` from `16:0/22:6` since both sum to 38:6.

### Data formats
- **METASPACE**: public web service hosting the LBA data, does annotation + serves ion images.
- **zarr**: chunked on-disk arrays, for huge imaging data.
- **parquet**: columnar table format.
- **AnnData**: what I actually compute on. `.X` (pixels x features), `.obs` (per-pixel metadata), `.var` (per-feature metadata), `.obsm` (embeddings/coords), `.layers`, `.uns`. Subsetting pixels keeps `.X` and `.obs` aligned automatically; `.var` doesn't change.

### What I actually do
1. Pull a section with the `metaspace` API (anonymous, public project). Settings: database CoreMetabolome v3, **FDR 0.1**, raw intensities (`scale_intensity=False`), monoisotopic peak only.
   - FDR 0.1 = I'm OK with ~1 in 10 annotations being wrong. METASPACE estimates it with **decoy formulas** that can't be real molecules.
2. `ds.results()` → one row per ion (formula + adduct + m/z). `ds.all_annotation_images()` → one 2D image per ion. Flatten + stack → pixels x ions.
3. Plot the mean spectrum (stick plot). Tallest peaks crowd around m/z 700 to 900, which is the phospholipid / sphingolipid zone.
4. **Adducts:** in positive mode a lipid shows up as [M+H]+, [M+Na]+, [M+K]+, [M+NH4]+. So one lipid = several peaks. Features are keyed as `formula__adduct`, e.g. `C45H91N2O6P__+H`.
5. Do both sections, apply the **tissue mask** (`data/masks/`), keep only ions found in both.
6. Join the **provided registration table** (CCF coords `xccf/yccf/zccf`, `acronym`, `allencolor`) on `(SectionID, x, y)`.
7. Save `01_raw.h5ad`.
8. First maps: an SM-range ion (two nitrogens in the formula = SM signature) lights up white matter; a neutral lipid ion likes grey matter. Also an **RGB overlay** of three ions to see co-localization.

**Sanity rule I should keep:** a real lipid map is spatially coherent (smooth territories). If it looks like speckle, suspect the data.

---

## 02 · From m/z peaks to lipid names (annotation)

**Goal:** turn each `formula__adduct` column into a defensible lipid name.

### Key ideas
- **A peak is a mass, not a lipid.** The instrument knows mass super precisely but has zero idea about identity.
- **Adducts** (Da added to neutral mass):

| adduct | added mass |
|---|---|
| [M+H]+ | 1.007276 |
| [M+Na]+ | 22.989769 |
| [M+K]+ | 38.963707 |
| [M+NH4]+ | 18.033823 |

  - Example: PC 34:1 (neutral 759.5778) shows up at four masses from ~760 to ~798.
  - This dataset is **dominated by [M+K]+** (lots of potassium in brain).
- **De-ionization:** `neutral = observed − adduct`. Databases store neutral masses.
- **ppm error:** `1e6 * |observed − reference| / reference`. Window used: **5 ppm** (same as the paper). At m/z 800 that's only **0.004 Da**. It's proportional, so wider in Da at high mass.

### The worked example (good to remember)
- Peak m/z **518.2643**, METASPACE says [M+K]+.
- Two LC-MS reference hits inside 5 ppm: **LPC 15:1 [K]** and **LPE 18:1 [K]**. Both ~0.01 ppm off. That's an **isobaric tie**: same formula (C23H46NO7P), identical mass.
- My simple matcher (top hit by ppm) picks **LPC 15:1**.
- But biology says otherwise: bulk quantitative LC-MS shows LPE 18:1 at ~0.02 nmol% and LPC 15:1 **not detected**. Odd-chain LPCs are basically absent in mammalian brain (animal fatty acids are mostly even-chain).
- The paper's tie-breaker is the **80% rule**: normalize candidates' bulk molar fractions, keep the top one if it's > 80% of the total. In EUCLID that's `Preprocessing.abundance_prioritization_lcms(..., threshold=0.8)`.
- The course deliberately keeps the simpler mass-only matcher, so a few names will differ from the paper. That's on purpose.

### Database layer
- **LIPID MAPS** (`structures.sdf`) has every known lipid structure + neutral exact mass. It also lands on LPC 15:1, so databases **can't break isobaric ties** either.
- Databases = breadth. LC-MS = confidence (actually seen and quantified in brain).
- Full paper pipeline scores trust by summing weights across references: LC-MS/MS = 8, ESI LC-MS = 2, published study = 1, METASPACE = 1 if FDR < 0.05 else 0.5.

### Annotating everything
- Loop over all 104 ions: top LC-MS hit within 5 ppm, else placeholder `ion_<mz>`.
- Result: **~63 of 104 get a real name**, ~1/3 of those had a silent isobaric tie. The rest stay `ion_...` (still real signal, just unnamed).
- Regex parser pulls out class, carbons, double bonds, ether flag (`O-` / `P-`). Caveat: for `SM 18:1;O2/24:1` the simple regex grabs only 18:1; the true sum is 42:2. The paper uses **goslin** for robust parsing.
- Save `02_annotated.h5ad`.

---

## 03 · Normalization with uMAIA

**Goal:** make the same lipid comparable across the two sections.

> Runs on the **`cajal-umaia`** kernel, JAX pinned to CPU.

> ⚠️ **Big caveat the notebook flags itself:** batch correction with 1 control vs 1 pregnant section is ill-defined. Any shift could be technical *or* biological (pregnancy, or just individual differences). Properly this is done at atlas scale (e.g. 6 sections x 4 animals x 2 conditions). It's an "intentional mistake" for teaching.

### Batch effect, what drifts between acquisitions
Matrix crystallization, laser energy, detector gain / ionization efficiency, sample handling. None of it is biology, all of it shifts intensity roughly section-wide.

### The empirical fact uMAIA relies on
- Per lipid, per section, the **log-intensity histogram is bimodal**:
  - low **background** mode (lipid absent, matrix noise), roughly fixed across slides → **the anchor**
  - higher **foreground** mode (lipid actually present) → **this is where batch drift happens**
- Mental model: **anchor low, drift high.** Correction should hold the background and slide the foreground back into register.
- Example: `HexCer 42:2` has a spike at the bottom (~1/3 of pixels) and a broad hump (white matter). Between sections the foreground medians are off by ~0.2 to 0.4 log units.

### The uMAIA model
Two-component Gaussian mixture in log space for every (lipid, section). The key line:

```
foreground mean  mu1_ac = locs_c  +  gamma_a * lambda_c  +  delta_c  (+ small error_ac slack)
                           anchor      batch shift            biological gap
```

- `locs_c`: background anchor, **one per lipid** (shared across sections)
- `gamma_a`: how much drift **this slide** has
- `lambda_c`: how sensitive **this lipid** is to drift
- `delta_c`: the biological fg-bg gap, should not be touched
- The batch shift is **rank-1** (slide factor x lipid factor). Paper justifies it with SVD: the first singular value dominates the technical error matrix.
- Normalization removes `gamma_a * lambda_c` (+ the slack) and keeps the anchor + `delta_c`.

**Fitting:** SVI with an `AutoDelta` guide = **MAP point estimate**, not posterior sampling. No MCMC. Adam optimizer, discrete labels marginalized out.

### The three calls
1. `uMAIA.norm.initialize(x, mask, subsample=True)` → GMM init (1 vs 2 components, pick by BIC)
2. `uMAIA.norm.normalize(...)` → the MAP fit, `num_steps=2000`, `seed=42`, **`covariate_vector=None`** (I don't tell it which section is pregnant, otherwise it could absorb the biology I want to test). Subsamples ~2,500 pixels/section, couple of minutes on CPU.
3. `uMAIA.norm.transform(x, mask, params)` → apply the correction

Input tensor shape: `(N_pixels, S_sections, V_molecules)` + a boolean mask (sections padded to equal length), log with `eps = 2e-4`.

### Result
- Median 90th-percentile gap between sections: **~0.21 → ~0.15 log units** (roughly a quarter to a third smaller). Exact numbers wobble a bit run to run because the subsample isn't fully seeded.
- Not zero, and it shouldn't be: with two sections uMAIA leaves real biology in rather than over-correcting.

### Unrolling the transform (histogram matching)
The correction is just `x → F⁻¹(G(x))`:
- `G` = this section's fitted mixture CDF (value → quantile)
- `F⁻¹` = shared reference inverse CDF (quantile → value). Reference params = **mean across sections** of the fitted params.
- Monotone, so pixel ranks within a section never change. Because G and F are mixtures, the stretch can differ for fg vs bg.
- Trick: `interp1d(cdf_vals, domain)` builds the inverse CDF by swapping x and y.

### Second, separate normalization: per-lipid 0 to 1 (`min01_per_lipid`)
- MALDI is **semi-quantitative**: ionization efficiency varies wildly by molecule, so raw intensities aren't comparable *across lipids*.
- Fix: clip at 0.5th / 99.5th percentile, rescale each lipid to [0, 1].

> **Two rules to carry forward:**
> - uMAIA = same lipid comparable **across sections**. min01 = different lipids comparable **to each other**. Both needed.
> - **Differential testing always runs on uMAIA-normalized, non-Harmonized data.**

Saves `03_normalized.h5ad` with `layers["umaia"]` (native, un-logged scale). Raw `X` stays untouched.

There's also a quick **composite sphingolipid score** (z-score each HexCer/Cer/SM, average), previewing the myelination score from NB07.

---

## 04 · Anatomy: registration and the Allen atlas

**Goal:** understand how each pixel gets a brain region address.

### Allen CCFv3
- 3D box of 25 µm voxels; any point = `(x, y, z)` in mm along AP, DV, ML.
- Plus an **annotation volume**: each voxel labeled with one of ~1000 hierarchical regions (acronym like `CP` = caudoputamen, plus an official color).
- So: coordinate first, region is just a lookup.

### Registration = affine + nonlinear warp
- **Affine** `p' = A p + t`: rotation, isotropic scale, anisotropic scale (undo shrinkage), shear, translation. Keeps lines straight and parallel. Can't fix local folds.
- **Nonlinear warp**: smooth displacement field, ideally a diffeomorphism (no tearing/folding). elastix splines (ABBA) or LDDMM (STalign).

### ABBA (the one thing that's pre-computed)
- Aligning Big Brains and Atlases, Fiji/QuPath plugin from EPFL's bioimaging platform. Steps: DeepSlice pre-alignment → affine elastix → spline elastix → BigWarp manual landmarks for stubborn slices. Also corrects **slicing angle**.
- Shipped ready-made because it's Java/C++, GUI-driven, flaky headless, and it's setup infrastructure, not the science.

### What I do
- Load `03_normalized.h5ad`, promote `layers["umaia"]` to `.X`.
- Voxel index = `floor(ccf_mm * 40)` (1 mm = 40 voxels at 25 µm).
- Slicing angle check: control sits at **one AP level**; pregnant spans **a few dozen AP levels over ~1 mm**, i.e. it was cut at a tilt and ABBA accounted for it. Both still center near AP 6.5.
- Region map with `spatial_categorical(..., color_key="allencolor")`. **174 distinct Allen regions** across both sections. Biggest: CP, then PIR, medial amygdala, a hippocampal field.
- Region **contours**: a pixel is on an edge if any neighbor has a different region. That's literally all `allen_contours` does.
- **Grey vs white matter** trick: every fiber-tract region in the Allen palette is `#cccccc`. So white matter = `allencolor == "#cccccc"`. (Ventricles `#aaaaaa`, unassigned `#ffffff`.) About **1 in 10 pixels is white matter**.
- `HexCer 42:2` is several times brighter in white matter and traces the tracts on its own. `marker_barplot` ranking (log2 mean-in-WM / mean-in-rest) puts sphingolipids at the top.
- **Region x lipid matrix** = one `groupby("acronym").mean()`. `sorted_lipid_heatmap` adds per-lipid 0-1 scaling + hierarchical clustering on cosine distance with optimal leaf ordering. This table powers the region-level differential test and the gene join later.

---

## 05 · Embedding: 104 lipids → a handful of programs

**Pipeline:** Moran's I feature selection → NMF → t-SNE (just to look) → Harmony (only for clustering)

Embedding is built on uMAIA values, min01-scaled per lipid. Learned on **control only**, then applied to both.

### Moran's I (spatial autocorrelation)
1. kNN graph in space, k = 6 (`cKDTree.query(coords, k=7)`, drop self)
2. center values
3. sum products over neighbor pairs
4. normalize: `I = (N/W) · Σ w_ij (x_i − x̄)(x_j − x̄) / Σ (x_i − x̄)²`

~1 = neighbors look alike (real anatomy), ~0 = salt and pepper.
- `HexCer 42:2` sits near the top, Moran ~0.74.
- Some of the highest-Moran columns are **unnamed ions**. Unannotated ≠ noise.

**Permutation null:** shuffle a lipid's values across pixels (keeps the histogram, kills spatial structure), recompute Moran many times, empirical p = `(1 + #null ≥ obs) / (1 + B)`. The null hugs zero. Even the lowest real column (~0.135) beats its own null, so **0.4 is a strength threshold, not a significance threshold**.

Filter: keep Moran > 0.4 **in both sections** → about 2/3 survive (~64 to 66 columns).

### Lipid-lipid correlation heatmap
Before NMF, look at modules: `np.corrcoef`, distance `1 − |r|`, optimal-leaf ordering. Red diagonal blocks = lipids that rise together. Tightest pair r ~0.95.

### NMF
- `V ≈ W H`. `H` = programs x lipids (recipes), `W` = pixels x programs (activities). All non-negative.
- vs PCA: PCA components mix + and − signs ("more PC minus SM"), NMF programs are purely additive parts so they're readable as biology. The toy 4x3 example shows this directly.
- **Seeded NMF** (`seeded_nmf`): cluster lipids by correlation distance, pick the most central lipid per cluster as a seed, run sklearn NMF with `init="custom"`. More stable than random init.
- **12 programs** here (paper uses 16 on the full atlas). `ConvergenceWarning` at 400 iterations is harmless.
- Programs ranked by `W.var(axis=0)` (spatial spread). Check that recipe (H row) and map (W column) agree.
- Apply to all pixels: `apply_nmf` = `model.transform` with H fixed → `obsm["X_nmf"]` (189,011 x 12).

### t-SNE (openTSNE, perplexity 30)
- Always run on the embedding, never raw lipids (faster, denoised, avoids curse of dimensionality).
- **For looking only.** Distances between blobs, blob sizes, and axes are meaningless.
- Colored by region → islands follow anatomy. Colored by section → residual split inside blobs = batch effect.

### Harmony
- Works on `X_nmf`, batch = **SectionID**. Iterates soft-clustering + nudging each batch toward cluster centers. Much more aggressive than uMAIA.
- → `obsm["X_harmony"]`

> Harmony output is **only for clustering and label transfer.** Never for the differential test. With two sections, batch and condition are perfectly confounded, so Harmony would happily erase the pregnancy effect.

### Bonus: a first neural net
- `ml.predict_position`: StandardScaler + `MLPRegressor(256,128,64)`, predict in-plane CCF (`yccf`, `zccf`) from lipids alone, 25% held out.
- Uses only the **right hemisphere** so bilateral symmetry can't confuse mediolateral.
- r ≈ **0.95 (yccf)** and **0.92 (zccf)**. The lipidome encodes location.

Saves `05_embedded.h5ad` (`X_nmf`, `X_tsne`, `X_harmony`, plus `var["moran_i"]`, `var["selected"]`).

---

## 06 · Clustering and label transfer

**Goal:** turn the Harmonized embedding into **lipizones** (lipid-defined territories), then put pregnant pixels in the same vocabulary.

### kNN graph → Leiden
- Build a kNN graph in the 12-D Harmony space (by hand on 2,000 pixels first, k=15 → 30,000 edges).
- **Leiden** maximizes **modularity** (edges inside communities minus random expectation). Decides cluster number itself.
- Knobs: `n_neighbors` (how local) and `resolution` (how finely it cuts; higher = more clusters).
- `cl.embedding.leiden_clusters`: k = 40 graph + `leidenalg.ModularityVertexPartition`. Passing `resolution` switches to the tunable partition.
- Clustered on **control only**. Gives a couple dozen lipizones (the notebooks quote ~18 to 26 depending on the run).
- The real atlas doesn't crank resolution to get hundreds of lipizones; it does coarse Leiden then a **divisive splitter** (more robust to batch).

### Lipizones vs anatomy
- `lipizone_colors` orders clusters by centroid similarity so similar lipizones get similar colors.
- Shapes "rhyme" with Allen regions but don't match. **ARI ≈ 0.06**. Lesson: lipizones partially track anatomy but are their own organizing principle.
- **Reciprocal enrichment** (the LBA metric) instead of raw crosstab: enrichment of lipizone in region x enrichment of region in lipizone. Cancels the size bias. Lives in `multimodal.reciprocal_enrichment`. Colors clipped to 2nd/98th percentile, not log scale.
- Marker lipids per cluster (`marker_lipids`, one-vs-rest log2FC): white-matter lipizones are led by HexCer; others by PCs like PC 32:0, plus some `ion_` channels.

### Label transfer (control → pregnant)
- `KNeighborsClassifier(n_neighbors=15, weights="distance")` fit on control Harmony coords + labels, predict pregnant. Confidence = max class probability.
- Mean confidence **~86%**. Low-confidence pixels sit at territory boundaries.
- Only works because Harmony put both sections in one space (harmonized on section, not condition).
- `cl.embedding.knn_transfer` is the one-liner.
- Extra views: hierarchy at 2/4/8/16 levels, highlight one lipizone in red, zoomed mosaic.

### EUCLID for real
- **EUCLID** = Enhanced uMAIA for Clustering Lipizones, Imputation, and Differential analysis. The package behind the atlas.
- Divisive top-down binary splitter: at each node, **local NMF** → k-means over-segment (K=15) → **backSPIN** re-aggregation to 2 groups → accept only if 3 gates pass (≥ a couple of differential lipids via Mann-Whitney, each half big enough, spatially coherent) → train an **XGBoost classifier per node**.
- Label transfer = walk each pixel root-to-leaf through the classifiers.
- Single plane → put all pixels in one "section" and bypass the rostrocaudal continuity gate.
- Deep tree on all control pixels: ~5 min, **~126 lipizones**, vs my handful-of-dozen Leiden ones. Big territories (cortex, white matter, deep nuclei) line up in both.

Saves `06_clustered.h5ad` with `obs["lipizone"]` (my Leiden + transfer) and `obs["euclid_lipizone"]`.

---

## 07 · Clustering from scratch + the pregnancy changes

Marked [WIP] in the repo. Same test used twice: once to validate clustering splits, once to find pregnancy changes.

### The differential test, built by hand
1. **Mann-Whitney U / Wilcoxon rank-sum** (same test, two names). Pool, rank, compare rank sums. Robust to skew and outliers, which MALDI data has plenty of.
2. **log2 fold change** = `log2(mean_B / mean_A)`. p tells me it's real, FC tells me it's big. With thousands of pixels, tiny shifts get tiny p.
3. **Benjamini-Hochberg** for 104 tests: sort p, compare `p(i)` to `(i/m)·α`, q-value = running min of `p(i)·m/i` from the top down. Controls FDR. In the toy, raw p = 0.04 fails after correction.

**Thresholds (atlas):** `|log2FC| > 0.2` (~15%) AND `q < 0.05`.

**Permutation null for a split:** a real cut on the top NMF program (grey/white) gives dozens of differential lipids; random label shuffles give ~0.

### Divisive splitter (simplified EUCLID)
Per node: local NMF (k=6) → standardize → KMeans(k=2) → gate (≥ 5 differential lipids, ≥ 2000 pixels) → recurse, max depth 3.
- Root splits 88,753 control pixels into ~75,000 / ~13,500, backed by ~86 differential lipids. Accepted splits carry 65 to 90 each → **8 leaf territories**.
- Painted with `allen_contours` on top: color blocks sit inside anatomical outlines.

### Pregnancy vs control
> Use `layers["umaia"]`, never `X_harmony`. Direction fixed as **log2(pregnant / control)**, positive = up in pregnancy.

- `analysis.differential_lipids` on all pixels: **~49 of 104** pass.
- Sphingolipids are mixed overall: SM 40:2 up (+0.32), one HexCer 42:2 channel up (+0.75), another HexCer 42:2 channel down (−0.23). Whole-section sphingolipid mean ≈ flat.
- That's not a contradiction with the paper. A whole-section average pools very different tissues.
- Volcano: x = log2FC, y = −log10(q), dashed lines at the thresholds. Plus violins for the top hits.

**Per-lipizone differential** (≥ 200 pixels per condition): 26 testable lipizones, change is **not uniform**. Some lipizones have dozens of lipids up (one has ~68), others barely move. This is the real pregnancy result: ask per territory.

### Reading territories
- Marker lipids per lipizone: HexCer/SM markers → white-matter-like; PC/PE → grey-matter-like.
- Sorted heatmaps by Allen region and by lipizone look broadly congruent → my unsupervised lipizones recover known anatomy.

### Composite scores
- **Membrane remodeling score** = sum of all log2FCs within a lipizone. Positive = net membrane building, negative = net turnover. Painted with a divergent map centered at 0.
- **Myelination score** = mean z-scored HexCer + Cer + SM per pixel. Traces white matter in both sections.

| view | control → pregnant |
|---|---|
| whole section | ~+0.007 → ~−0.006 (looks like nothing) |
| white matter only | **~+0.27 → ~+0.42** |
| per lipizone | ~14 of 26 go up, mostly the sphingolipid-rich ones |

> **Punchline:** the paper's headline (myelination signal up in pregnant white matter) does show up from my own lipizones, even though some individual lipids go the "wrong" way. Robust biology survives analysis details.

---

## 08 · Which genes explain the lipid changes (multimodal)

Also [WIP]. Goal: from *description* to a *mechanistic hypothesis*.

### The bridge
- MALDI (my data) and **Allen MERFISH** (different mice, different lab, imputed to 8460 genes) never touched the same tissue.
- Both are registered to **Allen CCF**, so they share coordinates and region names. That's the entire link.
- Two zoom levels:
  - **Per-pixel (fine):** borrow genes from nearby MERFISH cells
  - **Per-region (coarse):** join region-averaged genes to region-level lipid changes, then model

### Per-pixel integration
- `load_merfish_cells`: cells in my AP window, 500 measured genes (Ensembl transcript IDs), drops vascular + immune.
- `cKDTree` on cell coords, ball query per pixel → average genes + majority-vote cell type (`subclass`).
- Paper radius = 0.05 (50 µm), only a fraction of pixels match because MERFISH is discrete coronal sections with gaps. Course uses **0.1 (100 µm)** for better coverage, at the cost of blurring borders.
- Sanity check: borrowed **Mog** transcript (`ENSMUST00000102665`) and measured **HexCer 42:2** light up the same white matter.
- Cell-type territory map: oligodendrocytes on the tracts.
- Lipizone x cell-type reciprocal enrichment: blocky, lipid territories are often cell-type territories.

### Per-region modeling
1. **Region change matrix** (`region_change_matrix`): per Allen region, `log2(pregnant_mean / control_mean)` per ion, keep regions with ≥ 50 pixels in both → ~120 x 104.
2. **Join** with `avemerfish_imputed_named.parquet` (region x 8460 genes, real gene symbols) on acronym → ~100 to 110 shared regions.
3. Wide problem (~100 rows, 8460 features) → compress genes with **NMF into 20 gene programs** (MinMax each gene to [0,1] first).
4. **XGBoost**, one regressor per ion, 20 z-scored program activities as features. `max_depth=3`, 80% row/col subsampling, L2 penalty, small learning rate.
   - Boosting refresher: each new tree fits the residuals of the previous ones.
   - **No leakage:** MinMax, NMF, and z-scoring fit on **training regions only**.
   - Score = **held-out Pearson r**. A good fraction of ions get r > 0.3, mean is modest but positive.
   - The headline myelin sphingolipid is *not* well predicted, an honest negative.
5. **SHAP** (TreeSHAP via `pred_contribs=True`, drop last column = base value). Mean |SHAP| per program per ion → average across ions → program importance ranking.
6. **Leading genes** of the top program (`top_genes_for_program`, sort the H row). Often messy: a lipid regulator, some region markers, uncharacterized clones.
7. **Gene ontology**: `mygene` (symbols → Entrez), `goatools` with `go-basic.obo` + mouse `gene2go` (taxid 10090), Fisher exact + BH, background = all 8460 measured genes.
   - Scan the top 8 programs. The **most predictive** program and the **most interpretable** one aren't necessarily the same.
   - In a clean run, the clearest program is **myelination / ensheathment** (classic oligodendrocyte genes).
8. **Permutation null**: shuffle each ion's change across regions (keep genes fixed, break the claimed link), refit 20 times on the 12 best ions. Null sits near 0, observed is to the right.
9. **Figure**: export panels as vector PDFs, assemble by hand in Illustrator/Inkscape.

> **Interpretation guardrail:** this is correlation across regions, not causation. To claim cause I'd need a perturbation experiment.

---

## 09 · My own analysis

The scaffold comes off. I pick a question, tools, run it, defend it.

**Planning template** (fill before coding): question, what result would answer it, data, method, what would convince me I'm wrong, limits. Hardest box is "what result would answer it": if I can't picture the plot, the question is too vague.

**Worked direction A:** myelination score in white matter vs grey matter, control vs pregnant. Expect WM up in pregnant, GM flat. Confirms the *direction*, not the paper's effect size (they used 6 mice + a Bayesian model).

**Rules for using Claude Code:**
1. Read every line it writes.
2. Verify against reality (did the cell actually run? do shapes and signs make sense?).
3. Small steps, one cell at a time.
4. Give it context: `adata` structure, `obs['lipizone']`, the helpers in `src/`, and **two sections, no biological replicates** so it doesn't propose stats I can't run.

### Ideas I could do for my project
- Membrane remodeling per Allen region instead of per lipizone
- Compare my Leiden lipizones vs EUCLID lipizones in the pregnancy differential
- Look at ether lipids or unsaturation (double bonds) shifts by region
- Rerun NB08 with a different number of gene programs and see if the myelination program is stable
- Check how the annotation tie-breaking (80% rule) changes any of the downstream hits

---

## Golden rules (cross-cutting, the stuff that actually matters)

1. **A pixel is a patch, not a cell.**
2. **A peak is a mass, not a lipid** until annotated, and isobaric ties are real.
3. **uMAIA ≠ min01.** Across sections vs across lipids.
4. **Never test on Harmony.** Differential = uMAIA layer. Harmony = clustering + transfer only.
5. **Learn on control, transfer to pregnant.** One shared vocabulary.
6. **Whole-section averages hide things.** Ask per territory / per region.
7. **Permutation nulls** everywhere: Moran's I, splitter gate, XGBoost.
8. **p for reality, FC for size, BH for many tests.**
9. **t-SNE is for looking, not measuring.**
10. **Two sections = confounded batch and condition.** State it out loud.
11. Read the helper source before calling it.

---

## Quick glossary

- **MALDI-MSI**: matrix-assisted laser desorption/ionization mass spectrometry imaging
- **m/z**: mass-to-charge ratio
- **Adduct**: small ion stuck to a lipid to give it charge (H, Na, K, NH4)
- **Isobaric**: different molecules, same mass
- **ppm**: parts per million mass error
- **LC-MS / LC-MS/MS**: liquid chromatography + MS (orthogonal reference; MS/MS fragments for structure)
- **FDR**: false discovery rate
- **CCFv3**: Allen Common Coordinate Framework v3
- **ABBA**: Aligning Big Brains and Atlases (registration tool)
- **uMAIA**: the normalization method (Nature Methods 2025)
- **Moran's I**: spatial autocorrelation
- **NMF**: non-negative matrix factorization
- **Harmony**: embedding-level batch integration
- **Leiden**: graph community detection
- **Lipizone**: lipid-defined tissue territory
- **EUCLID**: the atlas's clustering / transfer / differential package
- **ARI**: Adjusted Rand Index (agreement between two partitions)
- **BH**: Benjamini-Hochberg FDR correction
- **MERFISH**: multiplexed error-robust FISH (spatial transcriptomics)
- **SHAP**: Shapley-value feature attribution
- **GO**: gene ontology
