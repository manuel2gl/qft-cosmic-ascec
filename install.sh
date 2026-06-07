#!/bin/bash
set -e  # Stop immediately if any command fails

#==========================================
# COSMIC ASCEC v04 ONE-CLICK INSTALLER
#==========================================
# Two ways to install:
#
#   (a) From GitHub (default):
#       wget https://raw.githubusercontent.com/manuel2gl/qft-cosmic-ascec/main/install.sh
#       bash install.sh
#
#   (b) From a local checkout (handy while developing):
#       cd /path/to/ascec_v04   # the dir holding install.sh
#       bash install.sh          # auto-detects local mode
#
# The script will:
#   - Get/refresh the source (clone from GitHub OR copy from local checkout)
#   - Install/configure Miniconda if needed
#   - Create a Python 3.11 conda env named 'py11' (or install into base)
#   - Install numpy, scipy, matplotlib, scikit-learn, cclib, openbabel, xtb
#   - Install orca-pi via pip (for ORCA 6.1+ output parsing)
#   - Set up `ascec` and `cosmic` shell aliases pointing at the root scripts
#
# Default annealing backend is xTB (installed from conda-forge). ORCA and
# Gaussian are optional and need to be installed separately if you want them.

#==========================================
# CONFIGURATION
#==========================================
# TRUE  -> create a separate 'py11' env with Python 3.11 (recommended)
# FALSE -> install into the base conda environment

INSTALL_PY11=TRUE

echo "> Starting COSMIC ASCEC v04 'One-Click' Installation..."

# -----------------------------------
# 1. Setup Directory & Download Code
#------------------------------------

TARGET_DIR="$HOME/software/ascec04"

# Detect "running from a local checkout": if the script's directory looks
# like the repo (has both ascec-v04.py and the cosmic_ascec/ package), copy
# from there instead of cloning. Lets the user iterate on local edits
# without needing to push to GitHub first.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_MODE=FALSE
if [ -f "$SCRIPT_DIR/ascec-v04.py" ] && [ -d "$SCRIPT_DIR/cosmic_ascec" ]; then
    LOCAL_MODE=TRUE
fi

# Ensure git is available — only needed on the clone/pull paths below. Tries
# the host's package manager (with sudo when not already root); if none works,
# bails with a clear message instead of a raw "git: command not found".
ensure_git() {
    if command -v git &> /dev/null; then
        return 0
    fi
    echo "> git not found — attempting to install it..."
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo &> /dev/null; then
            SUDO="sudo"
        else
            echo "  Cannot install git: not root and sudo unavailable."
            return 1
        fi
    fi
    if command -v apt-get &> /dev/null; then
        $SUDO apt-get update && $SUDO apt-get install -y git
    elif command -v dnf &> /dev/null; then
        $SUDO dnf install -y git
    elif command -v yum &> /dev/null; then
        $SUDO yum install -y git
    elif command -v zypper &> /dev/null; then
        $SUDO zypper install -y git
    elif command -v pacman &> /dev/null; then
        $SUDO pacman -Sy --noconfirm git
    elif command -v apk &> /dev/null; then
        $SUDO apk add git
    else
        echo "  No supported package manager found (apt/dnf/yum/zypper/pacman/apk)."
        return 1
    fi
    command -v git &> /dev/null
}

echo "> Setting up directories at $TARGET_DIR..."
mkdir -p "$TARGET_DIR"

if [ "$LOCAL_MODE" = "TRUE" ] && [ "$SCRIPT_DIR" != "$TARGET_DIR" ]; then
    echo "> Local checkout detected at $SCRIPT_DIR — copying into $TARGET_DIR..."
    # rsync preserves perms and skips junk; fall back to cp -a if rsync missing.
    if command -v rsync &> /dev/null; then
        rsync -a --delete \
            --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
            "$SCRIPT_DIR/" "$TARGET_DIR/"
    else
        cp -a "$SCRIPT_DIR/." "$TARGET_DIR/"
    fi
elif [ "$LOCAL_MODE" = "TRUE" ] && [ "$SCRIPT_DIR" = "$TARGET_DIR" ]; then
    echo "> Installing in place at $TARGET_DIR (no copy needed)..."
elif [ -d "$TARGET_DIR/.git" ]; then
    echo "> Repo exists, pulling latest updates..."
    ensure_git || { echo "> ERROR: git is required to pull updates. Install git and re-run."; exit 1; }
    cd "$TARGET_DIR" && git pull
else
    echo "> Cloning repository..."
    if ! ensure_git; then
        echo "-------------------------------------------------------"
        echo "> ERROR: git is required to clone the repository, and it"
        echo "> could not be installed automatically."
        echo "> "
        echo "> SOLUTIONS:"
        echo ">   - Install git manually, then re-run this script, OR"
        echo ">   - Run install.sh from a local checkout (a directory"
        echo ">     containing both ascec-v04.py and cosmic_ascec/), which"
        echo ">     copies the source instead of cloning (no git needed)."
        echo "-------------------------------------------------------"
        exit 1
    fi
    git clone https://github.com/manuel2gl/qft-cosmic-ascec.git "$TARGET_DIR"
fi

#-----------------------------------------
# 2. Check for Conda (Install if missing)
#-----------------------------------------

# Default install location if no existing conda is found
DEFAULT_MINICONDA_DIR="$HOME/miniconda3"

# Detect any existing conda installation (common locations)
MINICONDA_DIR=""
for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/conda" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -x "$candidate/bin/conda" ]; then
        MINICONDA_DIR="$candidate"
        break
    fi
done

if ! command -v conda &> /dev/null; then
    if [ -n "$MINICONDA_DIR" ]; then
        echo "> Conda installation found at $MINICONDA_DIR. Initializing..."
        eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"
        "$MINICONDA_DIR/bin/conda" init bash > /dev/null 2>&1
        if ! command -v conda &> /dev/null; then
            echo "> Updating existing conda installation at $MINICONDA_DIR..."
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
            bash miniconda.sh -b -u -p "$MINICONDA_DIR"
            eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"
            "$MINICONDA_DIR/bin/conda" init bash > /dev/null 2>&1
            rm miniconda.sh
        fi
    else
        echo "> Conda not found. Installing Miniconda to $DEFAULT_MINICONDA_DIR..."
        MINICONDA_DIR="$DEFAULT_MINICONDA_DIR"
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p "$MINICONDA_DIR"
        eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"
        "$MINICONDA_DIR/bin/conda" init bash > /dev/null 2>&1
        rm miniconda.sh
    fi
else
    echo "> Conda found at $(command -v conda). Proceeding..."
    eval "$(conda shell.bash hook)"
    if ! grep -q "conda initialize" "$HOME/.bashrc"; then
        echo "> Adding conda initialization to .bashrc..."
        conda init bash > /dev/null 2>&1
    fi
fi

#-----------------------------------------
# 3. Check Python Version Compatibility
#-----------------------------------------

PYTHON_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "> Detected Python $PYTHON_VERSION in base environment"

if [ "$INSTALL_PY11" = "FALSE" ]; then
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        echo "-------------------------------------------------------"
        echo "> ERROR: Python $PYTHON_VERSION is too old"
        echo "> COSMIC ASCEC requires Python >= 3.9 or <= 3.11"
        echo "> "
        echo "> SOLUTION: Set INSTALL_PY11=TRUE at the top of this script"
        echo "-------------------------------------------------------"
        exit 1
    fi
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
        echo "-------------------------------------------------------"
        echo "> ERROR: Python $PYTHON_VERSION is incompatible"
        echo "> COSMIC ASCEC requires Python >= 3.9 or <= 3.11"
        echo "> "
        echo "> SOLUTION: Set INSTALL_PY11=TRUE at the top of this script"
        echo "-------------------------------------------------------"
        exit 1
    fi
fi

#----------------------------------------------
# 4. Accept Conda Terms of Service
#----------------------------------------------

echo "> Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

#--------------------------
# 5. Environment Setup
#--------------------------

if [ "$INSTALL_PY11" = "TRUE" ]; then
    echo "> Creating/activating separate 'py11' environment with Python 3.11..."
    if conda env list | grep -q "^py11 "; then
        echo "> Environment 'py11' already exists. Activating..."
        conda activate py11
    else
        echo "> Creating new environment 'py11'..."
        conda create -n py11 python=3.11 -y
        conda activate py11
    fi
    echo "> Installing dependencies into 'py11' environment..."
else
    echo "> Using base environment for installation..."
fi

#----------------------------------------------
# 6. Install Dependencies
#----------------------------------------------

# Use libmamba solver — the classic conda solver can hang for 10+ minutes
# when solving conda-forge packages like openbabel/xtb. libmamba is the
# default in conda >= 23.10 but we set it explicitly for older installs.
conda install -n base -c conda-forge conda-libmamba-solver -y 2>/dev/null || true
conda config --set solver libmamba 2>/dev/null || true

# Single combined solve (one solver pass instead of two) with conda-forge
# as the primary channel. cclib parses ORCA<=5 / Gaussian output; openbabel
# writes .mol files; xtb is the default annealing backend.
#
# xtb is pinned to >=6.7: left unpinned, the combined solve can backtrack to
# the ancient conda-forge build 6.5.0 (2022), which crashes mid-optimization
# with a libgfortran I/O error and never writes a "final structure:" block.
# That makes cosmic extract zero geometries -> zero motifs -> "No motifs_/
# umotifs_ folder found" at the end of an otherwise green run. Pinning the
# floor forces a working build (or a loud solver error instead of silent
# breakage). 6.7.x is statically linked and runs cleanly.
conda install -c conda-forge --solver=libmamba -y \
    numpy scipy matplotlib scikit-learn cclib openbabel "xtb>=6.7"
# orca-pi parses ORCA 6.1+ structured property output. Optional: only used
# when ORCA 6.1+ is the QM backend; safe to skip on a pure-xtb workflow.
pip install orca-pi || echo "  (orca-pi install failed; ORCA 6.1+ parsing will fall back to text scrape)"

# Sanity-check that xtb is actually callable — the default annealing backend
# must be on PATH or annealing runs will fail with no useful error.
if command -v xtb &> /dev/null; then
    XTB_VER=$(xtb --version 2>&1 | grep -oP 'version \K[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
    echo "> xtb available: ${XTB_VER:-unknown}"
    # Guard against a too-old build slipping through (e.g. if the pin above was
    # relaxed). 6.5.0 and older conda-forge builds crash with a libgfortran I/O
    # error and produce no usable geometry, which silently breaks cosmic.
    XTB_MAJOR=$(echo "$XTB_VER" | cut -d. -f1)
    XTB_MINOR=$(echo "$XTB_VER" | cut -d. -f2)
    if [ -n "$XTB_MAJOR" ] && { [ "$XTB_MAJOR" -lt 6 ] || { [ "$XTB_MAJOR" -eq 6 ] && [ "$XTB_MINOR" -lt 6 ]; }; }; then
        echo "  WARNING: xtb $XTB_VER is too old and is known to crash mid-optimization."
        echo "  Reinstall a working build: conda install -c conda-forge 'xtb>=6.7'"
    fi
else
    echo "  WARNING: xtb not found on PATH after conda install."
    echo "  Try: conda install -c conda-forge 'xtb>=6.7'"
fi


#----------------------------------
# 7. Setup Shortcuts (Aliases)
#----------------------------------

echo "> Configuring shortcuts in .bashrc..."
BASHRC="$HOME/.bashrc"

# Remove any existing COSMIC ASCEC aliases before re-adding (ensures updates take effect)
sed -i '/# COSMIC ASCEC aliases/d' "$BASHRC"
sed -i '/alias ascec=/d' "$BASHRC"
sed -i '/alias cosmic=/d' "$BASHRC"

echo "" >> "$BASHRC"
echo "# COSMIC ASCEC aliases" >> "$BASHRC"
if [ "$INSTALL_PY11" = "TRUE" ]; then
    CONDA_BASE=$(conda info --base)
    ENV_BIN="$CONDA_BASE/envs/py11/bin"
    PYTHON_BIN="$ENV_BIN/python"
    # Prepend env bin to PATH so obabel, xtb and other env tools are found
    # at runtime. The aliases target the two root scripts; each script puts
    # TARGET_DIR on sys.path and calls into the cosmic_ascec package.
    echo "alias ascec='PATH=\"$ENV_BIN:\$PATH\" $PYTHON_BIN $TARGET_DIR/ascec-v04.py'" >> "$BASHRC"
    echo "alias cosmic='PATH=\"$ENV_BIN:\$PATH\" $PYTHON_BIN $TARGET_DIR/cosmic-v01.py'" >> "$BASHRC"
else
    echo "alias ascec='python $TARGET_DIR/ascec-v04.py'" >> "$BASHRC"
    echo "alias cosmic='python $TARGET_DIR/cosmic-v01.py'" >> "$BASHRC"
fi

echo "-------------------------------------------------------"
echo "> INSTALLATION COMPLETE!"
echo ">"
echo "> Reload your shell configuration:"
echo "    source ~/.bashrc"
echo ">"
echo "> Then use 'ascec' and 'cosmic' directly -- no environment activation needed."
echo ">"
echo "> Quick sanity check:"
echo "    ascec --version"
echo "    cosmic --version"
echo "-------------------------------------------------------"
