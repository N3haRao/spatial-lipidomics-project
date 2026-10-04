"""Sanity check: is this conda environment wired up to the tutorial and its data?

How to run it (once for each env, from the repo root):

    mamba run -n cajal-lipidomics python scripts/check_setup.py
    mamba run -n cajal-umaia      python scripts/check_setup.py

(scripts/setup.sh runs both of these automatically at the end.)

What it tells me:
  - which Python / env I'm actually running in (the #1 source of "ModuleNotFoundError")
  - whether my package and the tutorial's package import
  - whether the libraries THIS env needs import (the two envs need different things)
  - where the tutorial clone is, and which data bundle files are present

How to read the output:
  "ok " = good, "XX " = broken (with a hint after the arrow), "-- " = not there yet (not fatal).

Exit code is 0 if everything this env needs works, 1 otherwise. The exit code is what lets
setup.sh stop and complain when something's broken.
"""

# Lets me use newer type-hint syntax on any Python 3.
from __future__ import annotations

import importlib          # import a module by its name as a string, e.g. importlib.import_module("scanpy")
import sys                # tells me which Python is running (sys.executable / sys.prefix), and lets me set the exit code
from pathlib import Path  # just to grab the env name out of the Python's install folder

# Running tally: flips to False the first time any required check fails.
ok = True


def check(label: str, passed: bool, hint: str = "") -> None:
    """Print one ok/XX line, and if it failed, show a hint on how to fix it.

    `global ok` is needed because I'm changing the module-level `ok` from inside a function.
    Without it, Python would quietly create a new local variable instead.
    `ok &= passed` is short for `ok = ok and passed`: once anything fails, ok stays False.
    """
    global ok
    print(f"  {'ok ' if passed else 'XX '} {label}" + ("" if passed or not hint else f"   -> {hint}"))
    ok &= passed


def can_import(mod: str) -> bool:
    """Try importing a module by name. True if it works, False if it doesn't.

    I catch every Exception (not just ImportError) on purpose: some packages are installed
    but crash while importing (e.g. a numpy version clash raises an AttributeError deep
    inside). For a setup check, "installed but broken" should count as a failure too.
    """
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


# ---- which env am I in? -----------------------------------------------------------------
# sys.prefix is the folder this Python is installed in. For a conda env it ends in the env
# name, e.g. .../miniforge3/envs/cajal-lipidomics, so .name gives me "cajal-lipidomics".
env = Path(sys.prefix).name
is_umaia = env == "cajal-umaia"
print(f"\npython: {sys.executable}")   # the exact Python binary, handy for debugging kernels
print(f"env:    {env}")


# ---- my package + the tutorial's package ------------------------------------------------
# Both envs need both of these, because every notebook (including 03) imports them.
print("\nmy package + the tutorial's package")
check("lipid_project", can_import("lipid_project"), "pip install -e . (from my repo root)")
check("cajal_lipidomics", can_import("cajal_lipidomics"), "pip install -e external/lipidomics_tutorial_cajalcourse")


# ---- the libraries this specific env needs -----------------------------------------------
# The two envs have different jobs, so they need different libraries:
#   cajal-umaia      -> JAX / NumPyro / uMAIA for the normalization fit in notebook 03
#   cajal-lipidomics -> everything else. scanpy/anndata for the data container, openTSNE,
#                       harmonypy and leidenalg for embedding + clustering, xgboost/shap
#                       for the gene models, goatools for gene ontology, metaspace to pull data.
# Note "sklearn" is how scikit-learn is imported, and "metaspace" is the metaspace2020 package.
print("\nlibraries this env needs")
needed = (
    ["numpy", "anndata", "jax", "numpyro", "uMAIA"]
    if is_umaia
    else ["numpy", "pandas", "anndata", "scanpy", "sklearn", "openTSNE", "harmonypy",
          "leidenalg", "xgboost", "shap", "goatools", "metaspace"]
)
for mod in needed:
    check(mod, can_import(mod), "re-run bash scripts/setup.sh")


# ---- locations + data ---------------------------------------------------------------------
# Only possible if lipid_project imported, since that's the module that knows the paths.
if can_import("lipid_project"):
    from lipid_project import paths

    print("\nlocations")
    # The tutorial clone is required: no clone means no cajal_lipidomics and no data.
    check(f"tutorial clone   {paths.TUTORIAL_DIR}", paths.TUTORIAL_DIR.is_dir(),
          "bash scripts/setup.sh, or set LIPID_TUTORIAL_DIR")

    # The data bundle is a separate (big) download that I might have skipped with --no-data,
    # so missing files are shown with "--" as a heads up, but they don't fail the check.
    # These are the inputs the notebooks read, roughly in the order they need them:
    #   registration (NB01), masks (NB01), refs (NB02), go (NB08), MERFISH region averages (NB08)
    print("\nprovided input data (from the data bundle)")
    for rel in ["provided/registration_ccf.parquet", "masks", "refs", "go",
                "avemerfish_imputed_named.parquet"]:
        p = paths.tutorial_data(*rel.split("/"))   # "provided/x.parquet" -> ("provided", "x.parquet")
        print(f"  {'ok ' if p.exists() else '-- '} {rel}")
    print("  (anything marked -- : run  bash scripts/setup.sh  without --no-data)")
    print()
    paths.summary()   # the same overview the notebooks print in their first cell


# ---- friendly warning if I'm in the wrong env --------------------------------------------
# Classic mistake: running in "base" or some other env, where none of this is installed.
if env not in ("cajal-lipidomics", "cajal-umaia"):
    print("\n!! heads up: you're not in a course env. In VS Code, re-pick the kernel"
          " (cajal-lipidomics, or cajal-umaia for notebook 03).")


# ---- verdict ------------------------------------------------------------------------------------
print("\nALL GOOD" if ok else "\nSOMETHING'S MISSING (see XX lines above)")
# Exit code 0 = success, 1 = failure. setup.sh (with `set -e`) stops if this returns 1.
sys.exit(0 if ok else 1)
