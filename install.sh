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
#   (c) On a cluster, reusing the conda your site already provides:
#       module load anaconda3
#       CONDA_ROOT="$(conda info --base)" bash install.sh
#       # add CONDA_SOLVER=classic if that conda has no mamba and you would
#       # rather not wait for the script to probe for it
#
# The script will:
#   - Install/configure Miniconda if needed
#   - Get/refresh the source (clone from GitHub OR copy from local checkout)
#   - Create a Python 3.11 conda env named 'py11' (or install into base)
#   - Install numpy, scipy, matplotlib, cclib, openbabel, xtb -- with mamba
#     when it is available, and with classic conda when it is not
#   - Install orca-pi via pip (for ORCA 6.1+ output parsing)
#   - Set up `ascec` and `cosmic` shell aliases pointing at the root scripts
#
# The install is a git checkout, always. If git is missing the script installs
# it from conda-forge; if that fails it STOPS and tells you to install git
# yourself, rather than silently unpacking a tarball you cannot 'git pull' or
# 'git status' later. Machines that genuinely cannot have git can opt out with
# ALLOW_TARBALL=TRUE, which prints what it is giving up.
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
REPO_BRANCH="main"
REPO_TARBALL="https://github.com/manuel2gl/qft-cosmic-ascec/archive/refs/heads/${REPO_BRANCH}.tar.gz"

# TRUE -> allow the no-git tarball install (a plain directory: no 'git pull',
#         no 'git status', updates re-download the whole tree). Off by default
#         and meant only for machines with no way to get git at all. Can also
#         be passed per-run:  ALLOW_TARBALL=TRUE bash install.sh
ALLOW_TARBALL="${ALLOW_TARBALL:-FALSE}"

# Miniconda installer. To pin a reproducible version, replace the URL with a
# versioned installer from https://repo.anaconda.com/miniconda/ and set
# MINICONDA_SHA256 to the matching hash published at
# https://docs.anaconda.com/miniconda/. When MINICONDA_SHA256 is left empty the
# download is NOT verified and the script says so out loud.
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_SHA256=""

# Existing conda on a cluster or shared machine.
# Set CONDA_ROOT to the prefix of a conda you already have when it is not in
# one of the standard locations (module-provided installs usually are not):
#
#     module load anaconda3            # or whatever your site calls it
#     CONDA_ROOT="$(conda info --base)" bash install.sh
#
# The script then uses that conda instead of downloading Miniconda, and never
# writes into the shared prefix: the environment lands in your own envs dir
# (normally ~/.conda/envs) whenever the shared prefix is read-only.
CONDA_ROOT="${CONDA_ROOT:-}"

# Which package manager resolves the dependencies: auto | mamba | libmamba | classic
#
#   auto     -> mamba first, classic conda as the safety net. In order:
#                 1. a `mamba` binary already on PATH or in the conda prefix
#                 2. a conda that already carries the libmamba solver
#                    (the same solver engine, driven by conda -- this is the
#                    normal case for conda >= 23.10)
#                 3. install mamba, but only into a base this script owns
#                    (see INSTALL_MAMBA); conda-libmamba-solver if that fails
#                 4. classic conda -- slower, never broken
#   mamba    -> require mamba/libmamba; stop with an explanation if unavailable
#   libmamba -> require conda's libmamba solver specifically (no mamba binary)
#   classic  -> force classic conda and skip every probe above. Use this on an
#               old or locked-down conda where you already know mamba is absent.
#
# An existing conda that has neither mamba nor the libmamba plugin is left
# exactly as it is and driven with the classic solver. Nothing is installed
# into somebody else's base environment to make mamba appear.
CONDA_SOLVER="${CONDA_SOLVER:-auto}"

# May `mamba` be installed into the base environment in auto mode?
#   AUTO  -> only into a base this script installed itself (a fresh Miniconda).
#            An existing base -- yours or the cluster's -- is never touched.
#   TRUE  -> also allowed for an existing writable base. Note this can pull a
#            large re-solve of that base, which is why it is not the default.
#   FALSE -> never install mamba; use whatever is already there.
INSTALL_MAMBA="${INSTALL_MAMBA:-AUTO}"

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
USING_EXTERNAL_CONDA=FALSE
# TRUE only when this script downloads and installs Miniconda in this run. That
# base belongs to us, so it is the one place mamba may be added without asking.
BOOTSTRAPPED_CONDA=FALSE

if [ -n "$CONDA_ROOT" ]; then
    # An explicitly chosen conda wins over anything found on PATH, so a cluster
    # module install is used as-is instead of a second Miniconda being dropped
    # into $HOME.
    [ -x "$CONDA_ROOT/bin/conda" ] \
        || die "CONDA_ROOT=$CONDA_ROOT does not contain bin/conda." \
               "Point CONDA_ROOT at a conda *prefix* -- the directory that" \
               "'conda info --base' prints on that installation."
    info "Using the conda you selected: $CONDA_ROOT"
    MINICONDA_DIR="$CONDA_ROOT"
    USING_EXTERNAL_CONDA=TRUE
    eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"
else
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
            BOOTSTRAPPED_CONDA=TRUE
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
fi

have conda || die "conda is still not callable after bootstrapping." \
                  "Open a new shell and re-run this script."

CONDA_BASE=$(conda info --base)
CONDA_VERSION=$(conda --version 2>/dev/null | awk '{print $2}')
info "conda ${CONDA_VERSION:-unknown} at $CONDA_BASE"

# On a shared install the base prefix belongs to the site admins. Knowing this
# up front lets the script skip writes that would only fail, and explains why
# the new environment appears under ~/.conda/envs rather than next to base.
CONDA_BASE_WRITABLE=TRUE
[ -w "$CONDA_BASE/conda-meta" ] || CONDA_BASE_WRITABLE=FALSE
if [ "$CONDA_BASE_WRITABLE" = "FALSE" ]; then
    info "base env is read-only -- nothing will be installed into it."
    if [ "$INSTALL_PY11" != "TRUE" ]; then
        die "INSTALL_PY11=FALSE asks to install into the base environment," \
            "but $CONDA_BASE is not writable by you." \
            "" \
            "SOLUTION: set INSTALL_PY11=TRUE so a private '$ENV_NAME' env is" \
            "created in your own envs dir instead."
    fi
fi

#-----------------------------------------
# 1b. Choose the package manager
#-----------------------------------------
# mamba resolves conda-forge packages like openbabel/xtb in seconds where the
# classic solver can grind for 10+ minutes, so it is the preferred front-end.
# It is not a requirement though: `mamba` and conda's `--solver=libmamba` drive
# the same solver engine, and classic conda produces the same environment, only
# slower. This runs before the environment is created so that one tool handles
# creation, git and the dependency install alike.

PKG_CMD="conda"          # the binary that actually resolves packages
PKG_LABEL="classic conda"
SOLVER_ARGS=()           # conda-only; mamba does not take --solver

# Run a package subcommand (install/create) with the chosen tool and solver.
pkg_run() {
    local sub="$1"; shift
    "$PKG_CMD" "$sub" "${SOLVER_ARGS[@]}" "$@"
}

# mamba may be on PATH, or sitting in the conda prefix without the prefix being
# on PATH (common when conda was set up only through the shell hook).
find_mamba() {
    if have mamba; then
        command -v mamba
        return 0
    fi
    if [ -x "$CONDA_BASE/bin/mamba" ]; then
        echo "$CONDA_BASE/bin/mamba"
        return 0
    fi
    return 1
}

# conda >= 22.11 accepts --solver; before that the flag was --experimental-solver
# and there is no libmamba plugin worth chasing. Ask conda itself rather than
# parsing a version string.
conda_accepts_solver_flag() {
    conda install --help 2>/dev/null | grep -q -- '--solver'
}

# The plugin is imported by the python running conda, i.e. base's python.
conda_has_libmamba() {
    "$CONDA_BASE/bin/python" -c "import conda_libmamba_solver" >/dev/null 2>&1
}

use_mamba() {
    PKG_CMD="$1"
    PKG_LABEL="mamba"
    SOLVER_ARGS=()
}

use_libmamba() {
    PKG_CMD="conda"
    PKG_LABEL="conda + libmamba solver"
    SOLVER_ARGS=(--solver=libmamba)
}

use_classic() {
    PKG_CMD="conda"
    PKG_LABEL="classic conda"
    # Only pass the flag when conda understands it; an old conda has one solver
    # anyway and would just choke on an unknown option.
    if conda_accepts_solver_flag; then
        SOLVER_ARGS=(--solver=classic)
    else
        SOLVER_ARGS=()
    fi
}

# May we add mamba to the base env? Only a base this script created is ours to
# change; an existing one is left alone unless the user opts in explicitly.
may_install_mamba() {
    case "$INSTALL_MAMBA" in
        FALSE) return 1 ;;
        TRUE)  [ "$CONDA_BASE_WRITABLE" = "TRUE" ] && return 0; return 1 ;;
        AUTO)  [ "$BOOTSTRAPPED_CONDA" = "TRUE" ] && [ "$CONDA_BASE_WRITABLE" = "TRUE" ] \
                   && return 0; return 1 ;;
        *) die "INSTALL_MAMBA must be 'AUTO', 'TRUE' or 'FALSE'" \
               "(got '$INSTALL_MAMBA')." ;;
    esac
}

select_package_manager() {
    local mamba_bin=""

    case "$CONDA_SOLVER" in
        classic)
            use_classic
            info "Package manager: $PKG_LABEL (forced via CONDA_SOLVER=classic)."
            info "Expect the dependency solve to take several minutes."
            return 0
            ;;
        auto|mamba|libmamba) ;;
        *) die "CONDA_SOLVER must be 'auto', 'mamba', 'libmamba' or 'classic'" \
               "(got '$CONDA_SOLVER')." ;;
    esac

    # 1. An existing mamba wins outright -- nothing to install, fastest path.
    if [ "$CONDA_SOLVER" != "libmamba" ] && mamba_bin="$(find_mamba)"; then
        use_mamba "$mamba_bin"
        info "Package manager: mamba ($mamba_bin)."
        return 0
    fi

    # 2. conda already carrying the libmamba solver is just as fast.
    if conda_accepts_solver_flag && conda_has_libmamba; then
        use_libmamba
        info "Package manager: $PKG_LABEL (mamba binary not present, same solver engine)."
        return 0
    fi

    # 3. Try to obtain one, but only in a base we are allowed to touch.
    if may_install_mamba; then
        info "Installing mamba into the base env at $CONDA_BASE..."
        if conda install -n base -c conda-forge mamba -y >/dev/null 2>&1 \
           && mamba_bin="$(find_mamba)"; then
            use_mamba "$mamba_bin"
            info "Package manager: mamba (installed into base)."
            return 0
        fi
        warn "Installing mamba failed; trying the smaller conda-libmamba-solver."
        if conda_accepts_solver_flag \
           && conda install -n base -c conda-forge conda-libmamba-solver -y >/dev/null 2>&1 \
           && conda_has_libmamba; then
            use_libmamba
            info "Package manager: $PKG_LABEL."
            return 0
        fi
    elif [ "$CONDA_BASE_WRITABLE" != "TRUE" ]; then
        info "base env is read-only, so mamba cannot be added there."
    else
        info "This conda was already here, so its base env is left untouched."
        info "(INSTALL_MAMBA=TRUE would allow adding mamba to it.)"
    fi

    # 4. No mamba anywhere: classic conda still builds the same environment.
    if [ "$CONDA_SOLVER" = "mamba" ] || [ "$CONDA_SOLVER" = "libmamba" ]; then
        die "CONDA_SOLVER=$CONDA_SOLVER was requested, but neither a mamba binary" \
            "nor conda's libmamba solver is available in $CONDA_BASE," \
            "and this script is not allowed to install one there." \
            "" \
            "OPTIONS:" \
            "  1. Re-run with CONDA_SOLVER=classic -- same environment, slower solve." \
            "  2. Re-run with INSTALL_MAMBA=TRUE if that base is yours to change." \
            "  3. Ask your admin to run:" \
            "       conda install -n base -c conda-forge mamba" \
            "  4. Unset CONDA_ROOT and let this script install its own Miniconda" \
            "     into \$HOME, where it may set mamba up itself."
    fi

    use_classic
    warn "No mamba and no libmamba solver on this conda, and its base env is not"
    warn "ours to change -- falling back to $PKG_LABEL. The install still works"
    warn "and produces the same environment; the dependency solve is just slower,"
    warn "so allow several minutes before assuming it has hung."
}

select_package_manager

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
    # That env does not exist yet — this runs BEFORE step 4 creates it. Without
    # creating it here, `conda install -n py11` fails on every fresh machine and
    # the script used to drop silently to the tarball. Step 4 finds the env
    # already present and just activates it.
    if [ "$target_env" != "base" ] \
       && ! conda env list | awk '{print $1}' | grep -qx "$target_env"; then
        info "Creating the '$target_env' environment early so git can go into it..."
        pkg_run create -n "$target_env" python="$PY_VERSION" -y >/dev/null 2>&1 \
            || target_env="base"
    fi
    pkg_run install -n "$target_env" -c conda-forge git -y >/dev/null 2>&1 || return 1
    if [ "$target_env" = "base" ]; then
        export PATH="$CONDA_BASE/bin:$PATH"
    else
        export PATH="$CONDA_BASE/envs/$target_env/bin:$PATH"
    fi
    have git || return 1
    # This git lives inside the conda env, so it is on PATH here but NOT in the
    # user's normal shell. Say so, or `git status` in the install dir later
    # reports "command not found" and the checkout looks broken.
    warn "git was installed into the '$target_env' conda env, so it is only on"
    warn "PATH after 'conda activate $target_env'. For everyday use install it"
    warn "system-wide too:  $(git_install_hint)"
    return 0
}

# The command to install git by hand on this machine. Used in the error path:
# when git cannot be obtained we tell the user exactly what to type.
git_install_hint() {
    if   have apt-get; then echo "sudo apt install git"
    elif have dnf;     then echo "sudo dnf install git"
    elif have yum;     then echo "sudo yum install git"
    elif have pacman;  then echo "sudo pacman -S git"
    elif have zypper;  then echo "sudo zypper install git"
    elif have brew;    then echo "brew install git"
    else                    echo "install git using your system's package manager"
    fi
}

# git is a hard requirement unless the user explicitly opted into the tarball.
# Returning instead of dying lets callers that can still do something useful
# (a local copy already on disk) degrade to a warning.
require_git() {
    ensure_git && return 0
    # Explicit if, not `[ ... ] && return`: under `set -e` a compound list that
    # ends up false aborts the whole script when the caller is not a condition.
    if [ "$ALLOW_TARBALL" = "TRUE" ]; then
        return 1
    fi
    die "git is required, and installing it from conda-forge failed." \
        "" \
        "Install git yourself, then re-run this script:" \
        "  $(git_install_hint)" \
        "" \
        "Without git the install cannot be a repository: no 'git pull' to" \
        "update, no 'git status' to see your local edits." \
        "" \
        "If this machine truly cannot have git, opt into the plain-directory" \
        "install explicitly:" \
        "  ALLOW_TARBALL=TRUE bash $0"
}

# Turn a plain directory (tarball unpack, ZIP download, rsync copy) into a real
# checkout WITHOUT touching the files in it. The reset is the default --mixed
# kind: it writes .git/ and the index only, so local edits survive and simply
# show up as ordinary unstaged changes.
adopt_as_checkout() {
    if [ -d "$TARGET_DIR/.git" ]; then
        return 0
    fi
    have git || return 1
    info "$TARGET_DIR is not a checkout — adopting it as one (files untouched)..."
    if git -C "$TARGET_DIR" init -q \
        && git -C "$TARGET_DIR" remote add origin "$REPO_URL" \
        && git -C "$TARGET_DIR" fetch -q origin \
        && git -C "$TARGET_DIR" symbolic-ref HEAD "refs/heads/$REPO_BRANCH" \
        && git -C "$TARGET_DIR" reset -q "origin/$REPO_BRANCH" \
        && git -C "$TARGET_DIR" branch --set-upstream-to="origin/$REPO_BRANCH" \
               "$REPO_BRANCH" >/dev/null 2>&1; then
        info "Adopted — 'git status' and 'git pull' now work in $TARGET_DIR."
        info "Files that differ from origin/$REPO_BRANCH show as modifications."
        return 0
    fi
    # A half-built .git is worse than none: the next run would take the
    # "existing checkout" path and fail on 'git pull'.
    rm -rf "$TARGET_DIR/.git"
    warn "Could not adopt $TARGET_DIR as a checkout (no network for the fetch?)."
    return 1
}

# Single place that reports a non-repo install, so every path says the same
# thing instead of leaving the user to discover it from a git error.
warn_not_a_checkout() {
    warn "$TARGET_DIR is NOT a git checkout."
    warn "'git status' and 'git pull' will not work there, and this script will"
    warn "re-download instead of updating. To fix it later, install git"
    warn "($(git_install_hint)) and re-run this script — it will adopt the"
    warn "directory as a checkout without touching your files."
}

# Fetch a source tarball instead of cloning. Needs neither git nor a package
# manager, which matters on locked-down HPC and corporate machines. NOT a
# fallback any more: reachable only via ALLOW_TARBALL=TRUE, because what it
# leaves behind is a plain directory that no git command can work with.
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
    warn "Installed from a tarball (ALLOW_TARBALL=TRUE)."
    warn_not_a_checkout
    return 0
}

info "Setting up directories at $TARGET_DIR..."

if [ "$LOCAL_MODE" = "TRUE" ] && [ "$SCRIPT_DIR" = "$TARGET_DIR" ]; then
    info "Installing in place at $TARGET_DIR (no copy needed)..."
    mkdir -p "$TARGET_DIR"
    # A ZIP download or an older tarball install lands here: the code runs, but
    # the directory is not a repo. Repair that instead of leaving the user to
    # find out from `git status` months later. Only a warning if it cannot be
    # done — the tree is already on disk and working, so aborting helps nobody.
    if [ ! -d "$TARGET_DIR/.git" ]; then
        if ensure_git; then
            adopt_as_checkout || warn_not_a_checkout
        else
            warn "git is not available and could not be installed from conda-forge."
            warn_not_a_checkout
        fi
    fi

elif [ "$LOCAL_MODE" = "TRUE" ]; then
    info "Local checkout detected at $SCRIPT_DIR — copying into $TARGET_DIR..."
    mkdir -p "$TARGET_DIR"
    # rsync preserves perms and skips junk; fall back to cp -a if rsync missing.
    # rsync leaves any .git/ in the target alone (excluded, so --delete cannot
    # eat a real checkout); when there is none, one is created below.
    if have rsync; then
        rsync -a --delete \
            --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
            "$SCRIPT_DIR/" "$TARGET_DIR/"
    else
        cp -a "$SCRIPT_DIR/." "$TARGET_DIR/"
    fi
    if [ ! -d "$TARGET_DIR/.git" ]; then
        if ensure_git; then
            adopt_as_checkout || warn_not_a_checkout
        else
            warn "git is not available and could not be installed from conda-forge."
            warn_not_a_checkout
        fi
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
        # A non-empty, non-repo target is almost always an earlier tarball or
        # ZIP install. That is repairable in place, so try that before giving
        # up: adoption rewrites no working file.
        if ensure_git && adopt_as_checkout; then
            info "Continuing with the adopted checkout at $TARGET_DIR."
        else
            die "$TARGET_DIR already exists, is not empty, and is not a git checkout." \
                "" \
                "It could not be adopted as one either (git unavailable, or the" \
                "fetch from GitHub failed)." \
                "" \
                "SOLUTIONS:" \
                "  - Install git, then re-run this script — it will convert the" \
                "    directory into a checkout without touching your files:" \
                "      $(git_install_hint)" \
                "  - Or move/delete $TARGET_DIR, then re-run this script."
        fi
    # require_git dies with instructions unless ALLOW_TARBALL=TRUE, in which
    # case it returns 1 and the tarball path below runs.
    elif require_git; then
        info "Cloning repository..."
        git clone "$REPO_URL" "$TARGET_DIR"
    elif fetch_tarball; then
        info "Source installed without git."
    else
        die "Could not obtain the source." \
            "git is unavailable and the source tarball could not be downloaded." \
            "" \
            "SOLUTIONS:" \
            "  - Install git, then re-run this script:" \
            "      $(git_install_hint)" \
            "  - Or download the repository manually from" \
            "    https://github.com/manuel2gl/qft-cosmic-ascec and run" \
            "    install.sh from inside it."
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

# `conda tos` only exists on conda >= 24.11; older conda does not gate the
# defaults channel behind an acceptance at all. Probing first keeps an old
# cluster conda from printing a scary "invalid choice: 'tos'" here.
if conda tos --help >/dev/null 2>&1; then
    info "Accepting conda Terms of Service..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
else
    info "This conda has no 'tos' subcommand (pre-24.11) — nothing to accept."
fi

#--------------------------
# 5. Environment Setup
#--------------------------

if [ "$INSTALL_PY11" = "TRUE" ]; then
    info "Creating/activating separate '$ENV_NAME' environment with Python $PY_VERSION..."
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        info "Environment '$ENV_NAME' already exists. Activating..."
        conda activate "$ENV_NAME"
    else
        info "Creating new environment '$ENV_NAME' with $PKG_LABEL..."
        pkg_run create -n "$ENV_NAME" python="$PY_VERSION" -y
        # `conda activate` regardless of who created the env: mamba's own shell
        # activation needs `mamba shell hook`, which is not initialised here,
        # and conda activates a mamba-created env perfectly well.
        conda activate "$ENV_NAME"
    fi
    info "Installing dependencies into '$ENV_NAME' environment..."
    DEP_ENV="$ENV_NAME"
    # Ask conda where the env actually landed instead of assuming
    # $CONDA_BASE/envs/$ENV_NAME. When base is read-only -- the normal case for
    # a cluster module -- conda silently creates it under ~/.conda/envs, and the
    # assumed path would produce aliases pointing at a python that is not there.
    ENV_PREFIX="${CONDA_PREFIX:-$CONDA_BASE/envs/$ENV_NAME}"
    [ "$ENV_PREFIX" = "$CONDA_BASE/envs/$ENV_NAME" ] \
        || info "Environment prefix: $ENV_PREFIX"
else
    info "Using base environment for installation..."
    DEP_ENV="base"
    ENV_PREFIX="$CONDA_BASE"
fi

#----------------------------------------------
# 6. Install Dependencies
#----------------------------------------------

# The package manager (mamba or conda) and its solver flags were chosen back in
# step 1, before anything was created, so the same tool builds the environment
# and fills it. See select_package_manager().

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
info "Resolving dependencies with $PKG_LABEL..."
pkg_run install -n "$DEP_ENV" --override-channels -c conda-forge -y \
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
    ENV_BIN="$ENV_PREFIX/bin"
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
if [ "$USING_EXTERNAL_CONDA" = "TRUE" ]; then
    echo ">"
    echo "> This install reuses the conda at $CONDA_BASE."
    echo "> Dependencies were resolved with $PKG_LABEL."
echo "> The aliases point straight at $ENV_PREFIX/bin,"
    echo "> so they work in batch jobs without a 'module load' first. Add the"
    echo "> module line to your job script anyway if your site expects it."
fi
echo ">"
echo "> Quick sanity check:"
echo "    ascec --version"
echo "    cosmic --version"
echo "-------------------------------------------------------"
