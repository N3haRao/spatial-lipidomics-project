"""Where everything lives, so my notebooks never hard-code paths like "../../data/...".

Why this file exists
--------------------
The tutorial's notebooks use relative paths like "../../data/derived/01_raw.h5ad". That only
works if the notebook sits exactly two folders below the data. My notebooks live in a
different repo, so instead of sprinkling paths everywhere, I keep them all here. Every
notebook just does `from lipid_project import paths` and asks this module where stuff is.
If I ever move a folder, I fix it here once and every notebook keeps working.

The two places that matter
--------------------------
  TUTORIAL_DIR   My local clone of the CAJAL tutorial. It's NOT committed to my repo
                 (it lives in external/, which .gitignore skips). It gives me two things:
                   1. the `cajal_lipidomics` helper package (installed into my conda envs)
                   2. the "provided" input data from the tutorial's Zenodo bundle
                      (registration, tissue masks, LC-MS references, MERFISH, GO files)

  DERIVED        My own pipeline outputs: 01_raw.h5ad, 02_annotated.h5ad, and so on.
                 They live in THIS repo under data/derived/, which is also git-ignored
                 because the files are hundreds of MB each.

Pointing at a tutorial clone somewhere else
-------------------------------------------
By default I expect the tutorial at <repo>/external/lipidomics_tutorial_cajalcourse
(that's where scripts/setup.sh puts it). If I keep it somewhere else, I set an environment
variable in the terminal before starting Jupyter / VS Code:

    export LIPID_TUTORIAL_DIR=/path/to/lipidomics_tutorial_cajalcourse
"""

# Lets me write type hints like `tuple[str, ...]` and `list[Path]` on older Pythons too.
from __future__ import annotations

import os                 # only used to read the LIPID_TUTORIAL_DIR environment variable
from pathlib import Path  # Path objects are nicer than strings: `/` joins folders, .exists() checks them


# ---------------------------------------------------------------------------------------
# 1. Find the root of my repo
# ---------------------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Walk upward from this file until I find the folder that has pyproject.toml + notebooks/.

    Why not just hard-code it? Because the repo could be cloned anywhere (my laptop, a lab
    server, a different username). This file always sits at <repo>/src/lipid_project/paths.py,
    so walking up from here is guaranteed to hit the repo root no matter where the repo is.
    The leading underscore in the name is the Python convention for "internal helper, not
    meant to be called from outside".
    """
    # __file__ is the path of this .py file. .resolve() turns it into an absolute path and
    # follows any symlinks, so the walk-up below is reliable.
    here = Path(__file__).resolve()

    # .parents gives every folder above this file, nearest first:
    #   <repo>/src/lipid_project  ->  <repo>/src  ->  <repo>  ->  ...  ->  /
    for parent in here.parents:
        # The repo root is the first folder that has BOTH markers. Checking two things
        # (not just one) avoids accidentally matching some other project's pyproject.toml.
        if (parent / "pyproject.toml").exists() and (parent / "notebooks").is_dir():
            return parent

    # If I get here, the package was installed in some weird way (e.g. copied without the
    # repo around it). Better to fail loudly than to silently point at the wrong folder.
    raise RuntimeError("couldn't find the repo root (no pyproject.toml + notebooks/ above me)")


# Computed once, the moment the module is imported. Every other path is built off this.
REPO_ROOT = _find_repo_root()


# ---------------------------------------------------------------------------------------
# 2. Where the tutorial clone lives
# ---------------------------------------------------------------------------------------

# os.environ.get(name, default) reads an environment variable and falls back to the default
# if it isn't set. So: use LIPID_TUTORIAL_DIR if I set it, otherwise the standard location.
# .expanduser() turns a "~" into my home folder, .resolve() makes it absolute.
TUTORIAL_DIR = Path(
    os.environ.get("LIPID_TUTORIAL_DIR", REPO_ROOT / "external" / "lipidomics_tutorial_cajalcourse")
).expanduser().resolve()


# ---------------------------------------------------------------------------------------
# 3. The tutorial's provided inputs (I only ever READ these, never write to them)
# ---------------------------------------------------------------------------------------
# The data bundle unzips into <tutorial>/data/. These shortcuts point at its subfolders.

TUTORIAL_DATA = TUTORIAL_DIR / "data"
PROVIDED = TUTORIAL_DATA / "provided"   # registration_ccf.parquet: Allen CCF coords + region per pixel (NB01)
MASKS = TUTORIAL_DATA / "masks"         # tissue masks, True where there's brain and not empty glass (NB01)
REFS = TUTORIAL_DATA / "refs"           # LC-MS reference lists + LIPID MAPS structures.sdf (NB02)
GO = TUTORIAL_DATA / "go"               # gene ontology graph + mouse gene annotations (NB08)


# ---------------------------------------------------------------------------------------
# 4. My own outputs (these live in MY repo)
# ---------------------------------------------------------------------------------------

DATA = REPO_ROOT / "data"
DERIVED = DATA / "derived"              # the pipeline chain, each notebook saves its stage here (git-ignored)
FIGURES = REPO_ROOT / "figures"         # figures I want to keep and commit
RESULTS = REPO_ROOT / "results"         # small CSV tables I want to keep and commit

# The pipeline stage files, in order. Each notebook loads the previous stage and saves its own.
# Note there's no 04: notebook 04 (anatomy) only explores, it doesn't save a new stage.
# Giving them short names means I write paths.stage("normalized") instead of remembering
# whether it was "03_normalized.h5ad" or "03_normalised.h5ad" or whatever.
STAGES = {
    "raw": "01_raw.h5ad",                # NB01: pixels x raw ions, straight from METASPACE + Allen anatomy
    "annotated": "02_annotated.h5ad",    # NB02: same matrix, but ions now carry lipid names
    "normalized": "03_normalized.h5ad",  # NB03: adds layers["umaia"], the batch-corrected values
    "embedded": "05_embedded.h5ad",      # NB05: adds obsm X_nmf / X_tsne / X_harmony
    "clustered": "06_clustered.h5ad",    # NB06: adds obs["lipizone"] (+ EUCLID's version)
}


# ---------------------------------------------------------------------------------------
# 5. Little helper functions I call from notebooks
# ---------------------------------------------------------------------------------------

def tutorial_data(*parts: str) -> Path:
    """Build a path inside the tutorial's data/ folder.

    `*parts` means "any number of folder/file names", which get joined in order:
        tutorial_data("provided", "registration_ccf.parquet")
        -> <tutorial>/data/provided/registration_ccf.parquet
    Passing pieces separately (instead of one string with slashes) keeps it OS-independent.
    """
    return TUTORIAL_DATA.joinpath(*parts)


def stage(name: str) -> Path:
    """Path to one of my pipeline stage files, by its short name.

        stage("normalized")  ->  <repo>/data/derived/03_normalized.h5ad

    If I typo the name, I get a clear error listing the valid ones instead of a confusing
    "file not found" three cells later.
    """
    if name not in STAGES:
        raise KeyError(f"unknown stage {name!r}; pick one of {list(STAGES)}")
    return DERIVED / STAGES[name]


def ensure_dirs() -> None:
    """Create my output folders if they don't exist yet (safe to call over and over).

    data/derived/ especially: it's git-ignored, so on a fresh clone it simply isn't there,
    and writing an .h5ad into a missing folder would crash.
    parents=True also creates any missing folders above it; exist_ok=True means
    "don't complain if it's already there".
    """
    for d in (DERIVED, FIGURES, RESULTS):
        d.mkdir(parents=True, exist_ok=True)


def savefig(fig, name: str, formats: tuple[str, ...] = ("pdf", "png"), dpi: int = 200) -> list[Path]:
    """Save a matplotlib figure into figures/ in a couple of formats at once.

    - PDF is vector, so text stays editable when I assemble panels in Illustrator/Inkscape
      (that's how the tutorial wants the final multi-panel figure built in NB08).
    - PNG is what shows up nicely in the README or on GitHub.
    dpi only matters for the PNG (and for any rasterized scatter points inside the PDF).
    bbox_inches="tight" trims the extra white margin around the plot.

    Usage:  paths.savefig(fig, "myelination_map")
            -> figures/myelination_map.pdf and figures/myelination_map.png
    Returns the list of files it wrote, in case I want to print them.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = FIGURES / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        out.append(p)
    return out


def summary() -> None:
    """Print where everything points and whether it actually exists.

    I run this in the first cell of every notebook. If something's off (tutorial not cloned,
    data bundle not downloaded, previous stage not built yet), I see it right away instead
    of hitting a FileNotFoundError halfway down the notebook.
    "ok " means the thing exists, "-- " means it doesn't (yet).
    """
    def mark(p: Path) -> str:
        return "ok " if p.exists() else "-- "

    print(f"repo root      {REPO_ROOT}")
    print(f"{mark(TUTORIAL_DIR)}tutorial       {TUTORIAL_DIR}")
    print(f"{mark(TUTORIAL_DATA / 'provided')}provided data  {TUTORIAL_DATA}")
    print(f"{mark(DERIVED)}my outputs     {DERIVED}")

    # Which stages have I already built? This list comprehension keeps only the stage names
    # whose file exists in data/derived/, so I can tell at a glance how far along I am.
    done = [n for n, f in STAGES.items() if (DERIVED / f).exists()]
    print(f"stages built   {', '.join(done) if done else 'none yet'}")
