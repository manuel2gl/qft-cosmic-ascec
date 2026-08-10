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
#   - Install/configure Miniconda if needed
#   - Get/refresh the source (clone from GitHub OR copy from local checkout)
#   - Create a Python 3.11 conda env named 'py11' (or install into base)
#   - Install numpy, scipy, matplotlib, cclib, openbabel, xtb
#   - Install orca-pi via pip (for ORCA 6.1+ output parsing)
#   - Set up `ascec` and `cosmic` shell aliases pointing at the root scripts
#
# git is NOT a prerequisite: if it is missing the script installs it from
# conda-forge, and failing that falls back to downloading a source tarball.
#
# Default annealing backend is xTB (installed from conda-forge). ORCA and
# Gaussian are optional and need to be installed separately if you want them.

#==========================================
# CONFIGURATION
#==========================================
# TRUE  -> create a separate 'py11' env with Python 3.11 (recommended)
# FALSE -> install into the base conda environment

INSTALL_PY11=TRUE

TARGET_DIR="$HOME/software/ascec04"
ENV_NAME="py11"
PY_VERSION="3.11"
REPO_URL="https://github.com/manuel2gl/qft-cosmic-ascec.git"
REPO_TARBALL="https://github.com/manuel2gl/qft-cosmic-ascec/archive/refs/heads/main.tar.gz"

# Miniconda installer. To pin a reproducible version, replace the URL with a
# versioned installer from https://repo.anaconda.com/miniconda/ and set
# MINICONDA_SHA256 to the matching hash published at
# https://docs.anaconda.com/miniconda/. When MINICONDA_SHA256 is left empty the
# download is NOT verified and the script says so out loud.
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_SHA256=""

# Markers fencing our block in the user's shell rc file. Everything between
# them is ours to rewrite; everything outside is the user's and is never
# touched. (The previous version deleted any line matching 'alias ascec=',
# which would silently eat an unrelated alias of the same name.)
BLOCK_BEGIN="# >>> COSMIC ASCEC >>>"
BLOCK_END="# <<< COSMIC ASCEC <<<"

#==========================================
# HELPERS
#==========================================

info() { echo "> $*"; }
warn() { echo "  WARNING: $*" >&2; }

die() {
    echo "-------------------------------------------------------" >&2
    echo "> ERROR: $1" >&2
    shift
    for line in "$@"; do echo "> $line" >&2; done
    echo "-------------------------------------------------------" >&2
    exit 1
}

have() { command -v "$1" >/dev/null 2>&1; }

# Download URL -> FILE using whichever fetcher exists. Minimal images ship one
# or the other, rarely both, so requiring wget alone was a needless failure.
fetch() {
    local url="$1" out="$2"
    if have wget; then
        wget -q --show-progress -O "$out" "$url"
    elif have curl; then
        curl -fL# -o "$out" "$url"
    else
        die "Neither wget nor curl is available." \
            "Install one of them and re-run this script."
    fi
}

# Verify a downloaded file against $2 when a hash is configured.
verify_sha256() {
    local file="$1" expected="$2"
    if [ -z "$expected" ]; then
        warn "MINICONDA_SHA256 is not set — skipping checksum verification."
        warn "To pin it, see the CONFIGURATION block at the top of this script."
        return 0
    fi
    local actual=""
    if have sha256sum; then
        actual=$(sha256sum "$file" | awk '{print $1}')
    elif have shasum; then
        actual=$(shasum -a 256 "$file" | awk '{print $1}')
    else
        warn "No sha256sum/shasum available — cannot verify the download."
        return 0
    fi
    if [ "$actual" != "$expected" ]; then
        die "Checksum mismatch for $(basename "$file")." \
            "expected: $expected" \
            "actual:   $actual" \
            "Refusing to run an installer that does not match the pinned hash."
    fi
    info "Checksum verified."
}

# Rewrite our fenced block in a shell rc file, creating it if absent.
install_shell_block() {
    local rc="$1"
    [ -e "$rc" ] || touch "$rc"

    # Drop our fenced block if a previous install left one.
    sed -i "\|^${BLOCK_BEGIN}\$|,\|^${BLOCK_END}\$|d" "$rc"

    # Migrate installs that predate the fenced block. Only lines that are
    # unmistakably ours are removed: the old marker comment, and alias lines
    # that actually point into TARGET_DIR. An unrelated `alias ascec=...` the
    # user wrote themselves is left alone.
    sed -i '\|^# COSMIC ASCEC aliases$|d' "$rc"
    sed -i "\|^alias ascec=.*${TARGET_DIR}|d" "$rc"
    sed -i "\|^alias cosmic=.*${TARGET_DIR}|d" "$rc"

    # Collapse trailing blank lines, so repeated install/uninstall cycles do
    # not accumulate whitespace at the end of the user's rc file.
    sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$rc"

    {
        echo ""
        echo "$BLOCK_BEGIN"
        echo "# Managed by install.sh - edits between these markers are overwritten."
        if [ "$INSTALL_PY11" = "TRUE" ]; then
            # Prepend the env's bin to PATH so obabel, xtb and other env tools
            # resolve at runtime. The aliases target the two root scripts; each
            # puts TARGET_DIR on sys.path and calls into the cosmic_ascec package.
            echo "alias ascec='PATH=\"$ENV_BIN:\$PATH\" $PYTHON_BIN $TARGET_DIR/ascec-v04.py'"
            echo "alias cosmic='PATH=\"$ENV_BIN:\$PATH\" $PYTHON_BIN $TARGET_DIR/cosmic-v01.py'"
        else
            echo "alias ascec='python $TARGET_DIR/ascec-v04.py'"
            echo "alias cosmic='python $TARGET_DIR/cosmic-v01.py'"
        fi
        echo "$BLOCK_END"
    } >> "$rc"

    info "Updated $rc"
}

echo "> Starting COSMIC ASCEC v04 'One-Click' Installation..."

# Detect "running from a local checkout": if the script's directory looks like
# the repo (has both ascec-v04.py and the cosmic_ascec/ package), copy from
# there instead of cloning. Lets the user iterate on local edits without
# needing to push to GitHub first — and needs no git at all.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_MODE=FALSE
if [ -f "$SCRIPT_DIR/ascec-v04.py" ] && [ -d "$SCRIPT_DIR/cosmic_ascec" ]; then
    LOCAL_MODE=TRUE
fi

#-----------------------------------------
# 1. Check for Conda (Install if missing)
#-----------------------------------------
# Conda is bootstrapped BEFORE the source is fetched, because it doubles as the
# way we obtain git when the system does not have it (step 2). The Windows
# installer has always been ordered this way; this script now matches.

DEFAULT_MINICONDA_DIR="$HOME/miniconda3"

MINICONDA_DIR=""
for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/conda" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -x "$candidate/bin/conda" ]; then
        MINICONDA_DIR="$candidate"
        break
    fi
done

if ! have conda; then
    if [ -n "$MINICONDA_DIR" ]; then
        info "Conda installation found at $MINICONDA_DIR. Initializing..."
        eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
        "$MINICONDA_DIR/bin/conda" init bash > /dev/null 2>&1 || true
    else
        info "Conda not found. Installing Miniconda to $DEFAULT_MINICONDA_DIR..."
        MINICONDA_DIR="$DEFAULT_MINICONDA_DIR"
        # Download into a temp dir: the old version wrote miniconda.sh into the
        # current directory, which fails outright if the cwd is read-only.
        TMP_DIR="$(mktemp -d)"
        trap 'rm -rf "$TMP_DIR"' EXIT
        info "Downloading $MINICONDA_URL"
        fetch "$MINICONDA_URL" "$TMP_DIR/miniconda.sh"
        verify_sha256 "$TMP_DIR/miniconda.sh" "$MINICONDA_SHA256"
        bash "$TMP_DIR/miniconda.sh" -b -p "$MINICONDA_DIR"
        eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
        "$MINICONDA_DIR/bin/conda" init bash > /dev/null 2>&1 || true
        rm -rf "$TMP_DIR"
        trap - EXIT
    fi
else
    info "Conda found at $(command -v conda). Proceeding..."
    eval "$(conda shell.bash hook)"
    if ! grep -q "conda initialize" "$HOME/.bashrc" 2>/dev/null; then
        info "Adding conda initialization to .bashrc..."
        conda init bash > /dev/null 2>&1 || true
    fi
fi

have conda || die "conda is still not callable after bootstrapping." \
                  "Open a new shell and re-run this script."

CONDA_BASE=$(conda info --base)

#-----------------------------------------
# 2. Get/refresh the source
#-----------------------------------------
# git acquisition cascade — no sudo, no system package manager. The previous
# version shelled out to `sudo apt-get update && apt-get install git`, which
# prompts for a password mid-install and refreshes the whole apt index.

ensure_git() {
    have git && return 0
    info "git not found — installing it from conda-forge (no sudo needed)..."
    # Into the target env, NOT base: polluting the user's base environment can
    # force a full base re-solve and is hard to undo.
    local target_env="$ENV_NAME"
    [ "$INSTALL_PY11" = "TRUE" ] || target_env="base"
    conda install -n "$target_env" -c conda-forge git -y >/dev/null 2>&1 || return 1
    if [ "$INSTALL_PY11" = "TRUE" ]; then
        export PATH="$CONDA_BASE/envs/$ENV_NAME/bin:$PATH"
    else
        export PATH="$CONDA_BASE/bin:$PATH"
    fi
    have git
}

# Fetch a source tarball instead of cloning. Needs neither git nor a package
# manager, which matters on locked-down HPC and corporate machines.
fetch_tarball() {
    have tar || return 1
    local tmp
    tmp="$(mktemp -d)" || return 1
    info "Downloading source tarball (no git required)..."
    if ! fetch "$REPO_TARBALL" "$tmp/src.tar.gz"; then
        rm -rf "$tmp"; return 1
    fi
    if ! tar -xzf "$tmp/src.tar.gz" -C "$tmp"; then
        rm -rf "$tmp"; return 1
    fi
    local extracted
    extracted="$(find "$tmp" -maxdepth 1 -type d -name 'qft-cosmic-ascec-*' | head -1)"
    if [ -z "$extracted" ]; then
        rm -rf "$tmp"; return 1
    fi
    mkdir -p "$TARGET_DIR"
    cp -a "$extracted/." "$TARGET_DIR/"
    rm -rf "$tmp"
    warn "Installed from a tarball, so $TARGET_DIR is not a git checkout."
    warn "Future updates will re-download rather than 'git pull'."
    return 0
}

info "Setting up directories at $TARGET_DIR..."

if [ "$LOCAL_MODE" = "TRUE" ] && [ "$SCRIPT_DIR" = "$TARGET_DIR" ]; then
    info "Installing in place at $TARGET_DIR (no copy needed)..."
    mkdir -p "$TARGET_DIR"

elif [ "$LOCAL_MODE" = "TRUE" ]; then
    info "Local checkout detected at $SCRIPT_DIR — copying into $TARGET_DIR..."
    mkdir -p "$TARGET_DIR"
    # rsync preserves perms and skips junk; fall back to cp -a if rsync missing.
    if have rsync; then
        rsync -a --delete \
            --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
            "$SCRIPT_DIR/" "$TARGET_DIR/"
    else
        cp -a "$SCRIPT_DIR/." "$TARGET_DIR/"
    fi

elif [ -d "$TARGET_DIR/.git" ]; then
    info "Existing checkout found at $TARGET_DIR."
    if ! ensure_git; then
        warn "git is unavailable, so the existing checkout cannot be updated."
        warn "Continuing with the code already in $TARGET_DIR."
    # Never clobber local edits. A researcher who has been editing the code in
    # place is the normal case, and `git pull` on a dirty tree aborts the whole
    # script under `set -e`.
    elif [ -n "$(git -C "$TARGET_DIR" status --porcelain 2>/dev/null)" ]; then
        warn "Local modifications detected — SKIPPING 'git pull' to protect them:"
        git -C "$TARGET_DIR" status --porcelain | sed 's/^/      /'
        warn "Commit or stash these changes and re-run if you want the update."
    # --ff-only: git >= 2.27 refuses a plain pull on divergent branches, and a
    # merge is not something an installer should be starting on the user's behalf.
    elif git -C "$TARGET_DIR" pull --ff-only; then
        info "Updated to the latest revision."
    else
        warn "'git pull --ff-only' failed (diverged branch or no network)."
        warn "Continuing with the code already in $TARGET_DIR."
    fi

else
    # Target must be absent or empty for a clone. Cloning into a non-empty
    # directory fails with a raw git fatal, which is a confusing way to learn
    # that you unpacked a tarball here last month.
    if [ -d "$TARGET_DIR" ] && [ -n "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]; then
        die "$TARGET_DIR already exists, is not empty, and is not a git checkout." \
            "" \
            "SOLUTIONS:" \
            "  - Move or delete it, then re-run this script, OR" \
            "  - Run install.sh from a local checkout (a directory containing" \
            "    both ascec-v04.py and cosmic_ascec/), which copies the source" \
            "    instead of cloning."
    fi
    if ensure_git; then
        info "Cloning repository..."
        git clone "$REPO_URL" "$TARGET_DIR"
    elif fetch_tarball; then
        info "Source installed without git."
    else
        die "Could not obtain the source." \
            "git is unavailable, could not be installed from conda-forge, and" \
            "the source tarball could not be downloaded." \
            "" \
            "SOLUTIONS:" \
            "  - Install git with your package manager, then re-run, OR" \
            "  - Download the repository manually from" \
            "    https://github.com/manuel2gl/qft-cosmic-ascec and run" \
            "    install.sh from inside it (local-checkout mode needs no git)."
    fi
fi

[ -f "$TARGET_DIR/ascec-v04.py" ] || die "$TARGET_DIR/ascec-v04.py is missing after the source step." \
                                         "The download or copy did not complete correctly."

#-----------------------------------------
# 3. Check Python Version Compatibility
#-----------------------------------------
# Only relevant when installing into base — the py11 path creates its own
# interpreter. This probe used to run unconditionally with `grep -oP`, which is
# unavailable on BusyBox and, under `set -e`, aborted the whole install when it
# matched nothing.

if [ "$INSTALL_PY11" = "FALSE" ]; then
    PYTHON_VERSION=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
    [ -n "$PYTHON_VERSION" ] || die "No 'python' found in the base environment." \
                                    "Set INSTALL_PY11=TRUE at the top of this script."
    PYTHON_MAJOR=${PYTHON_VERSION%%.*}
    PYTHON_MINOR=${PYTHON_VERSION##*.}

    info "Detected Python $PYTHON_VERSION in base environment"

    # 3.10 is a hard floor: file_formats/asc_parser.py uses a `match` statement
    # (PEP 634), which is a SYNTAX error on 3.9 — and it sits in the .asc
    # parser, so a 3.9 install appears to succeed and then fails on first use.
    if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
        die "Python $PYTHON_VERSION is too old." \
            "COSMIC ASCEC requires Python 3.10 or newer (3.11 recommended)." \
            "" \
            "SOLUTION: Set INSTALL_PY11=TRUE at the top of this script."
    fi
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
        die "Python $PYTHON_VERSION is not supported yet." \
            "COSMIC ASCEC targets Python 3.10-3.12 (3.11 recommended);" \
            "conda-forge does not yet ship all dependencies for 3.13+." \
            "" \
            "SOLUTION: Set INSTALL_PY11=TRUE at the top of this script."
    fi
fi

#----------------------------------------------
# 4. Accept Conda Terms of Service
#----------------------------------------------

info "Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

#--------------------------
# 5. Environment Setup
#--------------------------

if [ "$INSTALL_PY11" = "TRUE" ]; then
    info "Creating/activating separate '$ENV_NAME' environment with Python $PY_VERSION..."
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        info "Environment '$ENV_NAME' already exists. Activating..."
        conda activate "$ENV_NAME"
    else
        info "Creating new environment '$ENV_NAME'..."
        conda create -n "$ENV_NAME" python="$PY_VERSION" -y
        conda activate "$ENV_NAME"
    fi
    info "Installing dependencies into '$ENV_NAME' environment..."
    DEP_ENV="$ENV_NAME"
else
    info "Using base environment for installation..."
    DEP_ENV="base"
fi

#----------------------------------------------
# 6. Install Dependencies
#----------------------------------------------

# Use libmamba — the classic solver can hang for 10+ minutes on conda-forge
# packages like openbabel/xtb. It is the default in conda >= 23.10; installing
# it covers older bases. Note we pass --solver=libmamba per command rather than
# running `conda config --set solver libmamba`, which would permanently rewrite
# the user's global ~/.condarc and is never undone by the uninstaller.
conda install -n base -c conda-forge conda-libmamba-solver -y 2>/dev/null || true

# Single combined solve (one solver pass instead of two), everything from
# conda-forge. --override-channels keeps `defaults` out of the solve entirely:
# mixing the two channels risks ABI mismatches between packages that share
# libraries. scikit-learn is deliberately absent — it was ~50 MB (plus joblib
# and threadpoolctl) for one StandardScaler z-score, which clustering/scaling.py
# now computes directly with numpy.
#
# cclib parses ORCA<=5 / Gaussian output; openbabel provides the obabel CLI;
# xtb is the default annealing backend.
#
# xtb is pinned to >=6.7: left unpinned, the combined solve can backtrack to
# the ancient conda-forge build 6.5.0 (2022), which crashes mid-optimization
# with a libgfortran I/O error and never writes a "final structure:" block.
# That makes cosmic extract zero geometries -> zero motifs -> "No motifs_/
# umotifs_ folder found" at the end of an otherwise green run. Pinning the
# floor forces a working build (or a loud solver error instead of silent
# breakage). 6.7.x is statically linked and runs cleanly.
conda install -n "$DEP_ENV" --override-channels -c conda-forge --solver=libmamba -y \
    numpy scipy matplotlib cclib openbabel "xtb>=6.7"

# orca-pi parses ORCA 6.1+ structured property output. Optional: only used
# when ORCA 6.1+ is the QM backend; safe to skip on a pure-xtb workflow.
pip install orca-pi || warn "orca-pi install failed; ORCA 6.1+ parsing will fall back to text scrape."

# Sanity-check that xtb is actually callable — the default annealing backend
# must be on PATH or annealing runs will fail with no useful error.
if have xtb; then
    XTB_VER=$(xtb --version 2>&1 | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p' | head -1)
    info "xtb available: ${XTB_VER:-unknown}"
    # Guard against a too-old build slipping through (e.g. if the pin above was
    # relaxed). 6.5.0 and older conda-forge builds crash with a libgfortran I/O
    # error and produce no usable geometry, which silently breaks cosmic.
    XTB_MAJOR=${XTB_VER%%.*}
    XTB_REST=${XTB_VER#*.}
    XTB_MINOR=${XTB_REST%%.*}
    if [ -n "$XTB_MAJOR" ] && [ "$XTB_MAJOR" -eq "$XTB_MAJOR" ] 2>/dev/null; then
        if [ "$XTB_MAJOR" -lt 6 ] || { [ "$XTB_MAJOR" -eq 6 ] && [ "${XTB_MINOR:-0}" -lt 6 ]; }; then
            warn "xtb $XTB_VER is too old and is known to crash mid-optimization."
            warn "Reinstall a working build: conda install -c conda-forge 'xtb>=6.7'"
        fi
    fi
else
    warn "xtb not found on PATH after conda install."
    warn "Try: conda install -c conda-forge 'xtb>=6.7'"
fi

#----------------------------------
# 7. Setup Shortcuts (Aliases)
#----------------------------------

info "Configuring shortcuts..."

if [ "$INSTALL_PY11" = "TRUE" ]; then
    ENV_BIN="$CONDA_BASE/envs/$ENV_NAME/bin"
    PYTHON_BIN="$ENV_BIN/python"
else
    ENV_BIN=""
    PYTHON_BIN="python"
fi

install_shell_block "$HOME/.bashrc"
# zsh is the default shell on macOS and increasingly common on Linux; the
# readme has always documented ~/.zshrc but the installer never wrote to it.
[ -f "$HOME/.zshrc" ] && install_shell_block "$HOME/.zshrc"

#----------------------------------
# 8. Verify the install actually works
#----------------------------------
# Fail loudly here rather than at the user's first real run.

info "Verifying installation..."
if [ "$INSTALL_PY11" = "TRUE" ]; then
    VERIFY_PYTHON="$PYTHON_BIN"
else
    VERIFY_PYTHON="python"
fi

if PATH="${ENV_BIN:+$ENV_BIN:}$PATH" "$VERIFY_PYTHON" "$TARGET_DIR/ascec-v04.py" --version; then
    info "Verification passed."
else
    die "'ascec --version' failed." \
        "The environment was created but the code does not run." \
        "Re-run this script, or report the error above as an issue."
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
