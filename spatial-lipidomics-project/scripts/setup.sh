#!/usr/bin/env bash
# =========================================================================================
# One-time setup: connects my repo to the CAJAL tutorial without copying any of its files in.
# =========================================================================================
#
# How to run it (from anywhere, but usually from the repo root):
#
#   bash scripts/setup.sh              # everything: clones, envs, kernels, ~1.3 GB data download
#   bash scripts/setup.sh --no-data    # same, but skip the data download for now
#
# It's safe to re-run. Every step first checks whether it's already done and skips it, so if
# something fails halfway (flaky wifi, a typo), I fix it and just run the script again.
#
# Works on Mac, Linux, and Windows through WSL2. Not native Windows: notebook 03 uses uMAIA,
# which needs JAX, and JAX has no native Windows build.
#
# What it does, in order:
#   1. clones the tutorial, uMAIA and EUCLID into external/  (git-ignored, never committed)
#   2. creates the two conda envs from the TUTORIAL's own environment files:
#        cajal-lipidomics  -> every notebook except 03
#        cajal-umaia       -> notebook 03 only (uMAIA needs an older JAX/numpy stack)
#   3. installs the tutorial's helper package (cajal_lipidomics) + my package (lipid_project)
#      into both envs
#   4. registers both envs as Jupyter kernels, so VS Code / Jupyter can pick them
#   5. downloads the tutorial's data bundle into external/lipidomics_tutorial_cajalcourse/data/
#   then runs scripts/check_setup.py in both envs to confirm everything works.
# =========================================================================================

# "Strict mode" for bash, so problems stop the script instead of being silently ignored:
#   -e           stop as soon as any command fails
#   -u           treat using an undefined variable as an error (catches typos in names)
#   -o pipefail  if any command in a pipe (a | b) fails, the whole pipe counts as failed
set -euo pipefail


# ---- read the optional --no-data flag -------------------------------------------------
# FETCH_DATA=1 means "download the data bundle". If the first argument is --no-data, flip it.
# "${1:-}" means "the first argument, or an empty string if there isn't one". Without the
# ":-" part, `set -u` would crash the script when I run it with no arguments.
FETCH_DATA=1
[[ "${1:-}" == "--no-data" ]] && FETCH_DATA=0


# ---- figure out where things are --------------------------------------------------------
# REPO = the root of my repo. BASH_SOURCE[0] is the path of this script (scripts/setup.sh),
# dirname strips the file name, "/.." goes up one folder, and `cd ... && pwd` turns it into
# an absolute path. This way the script works no matter which folder I run it from.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# EXT = where all the outside code gets cloned. It's in .gitignore, so none of it gets committed.
EXT="$REPO/external"

# TUT = the tutorial clone. Same override as in src/lipid_project/paths.py: if I've set
# LIPID_TUTORIAL_DIR, use that, otherwise the default spot inside external/.
TUT="${LIPID_TUTORIAL_DIR:-$EXT/lipidomics_tutorial_cajalcourse}"


# ---- pick the package manager -----------------------------------------------------------
# mamba is a faster drop-in replacement for conda (Miniforge ships with it). Use it if it's
# there, fall back to conda, and bail out with a helpful message if neither is installed.
# `command -v X` prints where X lives if it exists; >/dev/null 2>&1 hides that output,
# so I only use the success/failure result.
if command -v mamba >/dev/null 2>&1; then PM=mamba
elif command -v conda >/dev/null 2>&1; then PM=conda
else echo "!! neither mamba nor conda found. Install Miniforge first: https://github.com/conda-forge/miniforge"; exit 1
fi


# ---- small helper functions ----------------------------------------------------------------

# say: print a visible section header so I can follow along in the terminal output.
say() { printf '\n==> %s\n' "$*"; }

# env_exists <name>: true if a conda env with exactly that name already exists.
#   `$PM env list` prints one env per line, awk keeps the first column (the env name),
#   grep -qx checks for an exact, whole-line match (-x) and stays quiet (-q).
#   Exact match matters: I don't want "cajal-lipidomics" to match "cajal-lipidomics-old".
env_exists() { "$PM" env list | awk '{print $1}' | grep -qx "$1"; }

# in_env <env> <command...>: run a command INSIDE a conda env without activating it.
#   `mamba run -n env cmd` is the script-friendly way to do this. `conda activate` is
#   meant for interactive shells and is unreliable inside scripts.
#   `shift` drops the env name from the argument list, so "$@" is just the command.
in_env() { local env="$1"; shift; "$PM" run -n "$env" "$@"; }

# clone <url> <destination>: git clone, unless it's already been cloned there.
#   --depth 1 grabs only the latest snapshot, not the whole history. Faster and smaller,
#   and I don't need their history, just their current code.
clone() {
  if [[ -d "$2/.git" ]]; then echo "   already cloned: $2"; else git clone --depth 1 "$1" "$2"; fi
}

# Make sure the folders exist before anything writes into them.
# -p means "also create parents, and don't complain if it already exists".
mkdir -p "$EXT" "$REPO/data/derived"


# ---- 1. clone the outside code ----------------------------------------------------------
say "1/5  cloning the tutorial, uMAIA and EUCLID into external/"
clone https://github.com/lamanno-epfl/lipidomics_tutorial_cajalcourse.git "$TUT"   # helpers + data scripts
clone https://github.com/lamanno-epfl/uMAIA.git  "$EXT/uMAIA"                       # normalization (NB03)
clone https://github.com/lamanno-epfl/EUCLID.git "$EXT/EUCLID"                      # atlas clustering engine (NB06)


# ---- 2. create the conda envs ------------------------------------------------------------
say "2/5  creating conda envs (first time takes a while, this is normal)"
# I build the envs from the tutorial's OWN environment files rather than writing my own.
# Their files carefully pin versions (numpy<2, pandas<2.2, ...) because unpinned installs
# break scanpy/anndata, so reusing them is much safer.
#
# The `( cd "$TUT" && ... )` part runs inside a subshell (the parentheses), so the cd only
# applies there. It matters because environment.yml says `-r requirements-extra.txt`, and
# that relative path only resolves if I'm standing in the tutorial folder.
if env_exists cajal-lipidomics; then echo "   cajal-lipidomics already exists"
else (cd "$TUT" && "$PM" env create -f environment.yml); fi

if env_exists cajal-umaia; then echo "   cajal-umaia already exists"
else (cd "$TUT" && "$PM" env create -f environment-umaia.yml); fi


# ---- 3. install the packages into both envs -----------------------------------------------
say "3/5  installing cajal_lipidomics (tutorial) + lipid_project (mine) into both envs"
# `pip install -e <folder>` = "editable" install. Instead of copying the code into the env,
# pip just points the env at the folder. So if I edit src/lipid_project/paths.py (or pull
# updates to the tutorial), the change shows up right away, no reinstall needed.
# -q keeps pip quiet so the output stays readable.
in_env cajal-lipidomics pip install -q -e "$TUT"     # the tutorial's helpers -> `import cajal_lipidomics`
in_env cajal-lipidomics pip install -q -e "$REPO"    # my package            -> `import lipid_project`

# For the uMAIA env I add --no-deps. That env is pinned to an older stack for JAX, and I don't
# want pip "helpfully" upgrading numpy/pandas inside it to satisfy some other package. The
# tutorial's own setup guide does the same thing.
in_env cajal-umaia pip install -q -e "$TUT"        --no-deps
in_env cajal-umaia pip install -q -e "$REPO"       --no-deps
in_env cajal-umaia pip install -q -e "$EXT/uMAIA"  --no-deps   # uMAIA itself, only needed in this env


# ---- 4. register the Jupyter kernels -----------------------------------------------------
say "4/5  registering Jupyter kernels"
# A conda env isn't automatically visible to Jupyter / VS Code. ipykernel install --user
# writes a small "kernelspec" file in my home folder that says "there's a kernel called X,
# and it runs this env's Python". After this, both show up in the kernel picker by name.
# The --name values match the kernelspec already set inside my starter notebooks.
in_env cajal-lipidomics python -m ipykernel install --user --name cajal-lipidomics --display-name "cajal-lipidomics"
in_env cajal-umaia      python -m ipykernel install --user --name cajal-umaia      --display-name "cajal-umaia"


# ---- 5. download the data bundle ---------------------------------------------------------
if [[ "$FETCH_DATA" == 1 ]]; then
  say "5/5  downloading the tutorial data bundle (~1.3 GB, resumable if it gets cut off)"
  # The tutorial ships its own downloader. It writes into "data/" relative to wherever it's
  # run, so I cd into the tutorial first (inside a subshell again), and the data lands in
  # <tutorial>/data/, which is exactly where paths.TUTORIAL_DATA looks.
  # It uses curl with resume, so if it dies at 60% I just re-run and it picks up from there.
  (cd "$TUT" && in_env cajal-lipidomics python scripts/fetch_data_bundle.py)
else
  say "5/5  skipped the data download (--no-data). Re-run without the flag when ready."
fi


# ---- final check ---------------------------------------------------------------------------
say "checking both envs"
# Run my checker inside each env. Because of `set -e` at the top, if a check fails the script
# stops here with an error, which is what I want: I'll see exactly which line said XX.
in_env cajal-lipidomics python "$REPO/scripts/check_setup.py"
in_env cajal-umaia      python "$REPO/scripts/check_setup.py"

say "all set. Open notebooks/01_mass_spectra_and_data.ipynb and pick the cajal-lipidomics kernel."
