#!/bin/bash
set -e  # Stop immediately if any command fails

#==========================================
# COSMIC ASCEC v04 UNINSTALLER (Linux)
#==========================================
# Reverses what install.sh set up, WITHOUT touching conda itself:
#   - Removes the 'py11' conda environment (leaves base conda alone)
#   - Removes the cloned/copied source directory ($HOME/software/ascec04)
#   - Removes the `ascec` and `cosmic` aliases from ~/.bashrc and ~/.zshrc
#
# Conda (Miniconda/Anaconda) is intentionally left installed, since the user
# may rely on it for other work. Run with --yes to skip the confirmation prompt.

ENV_NAME="py11"
TARGET_DIR="$HOME/software/ascec04"

# Must match install.sh.
BLOCK_BEGIN="# >>> COSMIC ASCEC >>>"
BLOCK_END="# <<< COSMIC ASCEC <<<"

ASSUME_YES=FALSE
for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=TRUE ;;
    esac
done

info() { echo "> $*"; }
warn() { echo "  WARNING: $*" >&2; }

echo "> COSMIC ASCEC v04 uninstaller"
echo ">"
echo "> This will:"
echo ">   - Remove the conda environment '$ENV_NAME' (conda itself is kept)"
echo ">   - Remove the source directory $TARGET_DIR"
echo ">   - Remove the 'ascec'/'cosmic' aliases from your shell rc files"
echo ">"

if [ "$ASSUME_YES" != "TRUE" ]; then
    read -r -p "> Proceed? [y/N] " reply
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "> Aborted."; exit 0 ;;
    esac
fi

# -----------------------------------
# 1. Remove the conda env (not conda)
# -----------------------------------

# Detect conda the same way install.sh does, so we can call it even if the
# current shell hasn't been `conda init`-ed. CONDA_ROOT matches the installer's
# escape hatch for a cluster/module conda that lives outside the usual places:
#     CONDA_ROOT="$(conda info --base)" bash uninstall.sh
if [ -n "${CONDA_ROOT:-}" ] && [ -x "$CONDA_ROOT/bin/conda" ]; then
    eval "$("$CONDA_ROOT/bin/conda" shell.bash hook)"
elif ! command -v conda &> /dev/null; then
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/conda" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
        if [ -x "$candidate/bin/conda" ]; then
            eval "$("$candidate/bin/conda" shell.bash hook)"
            break
        fi
    done
fi

if command -v conda &> /dev/null; then
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        info "Removing conda environment '$ENV_NAME'..."
        # Deactivate first if we happen to be inside it, or removal fails.
        conda deactivate 2>/dev/null || true
        conda env remove -n "$ENV_NAME" -y
    else
        info "Conda environment '$ENV_NAME' not found — nothing to remove."
    fi
else
    warn "conda not found on PATH or in common locations."
    echo "  Skipping env removal. If '$ENV_NAME' exists, remove it manually:"
    echo "    conda env remove -n $ENV_NAME"
fi

# -----------------------------------
# 2. Remove our block from shell rc files
# -----------------------------------
# Only the fenced block and alias lines that actually point into TARGET_DIR are
# removed. An unrelated `alias ascec=...` the user wrote themselves survives.

remove_shell_block() {
    local rc="$1"
    [ -f "$rc" ] || return 0
    info "Removing aliases from $rc..."
    sed -i "\|^${BLOCK_BEGIN}\$|,\|^${BLOCK_END}\$|d" "$rc"
    # Clean up installs that predate the fenced block.
    sed -i '\|^# COSMIC ASCEC aliases$|d' "$rc"
    sed -i "\|^alias ascec=.*${TARGET_DIR}|d" "$rc"
    sed -i "\|^alias cosmic=.*${TARGET_DIR}|d" "$rc"
    # Collapse the trailing blank lines the removed block leaves behind.
    sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$rc"
}

remove_shell_block "$HOME/.bashrc"
remove_shell_block "$HOME/.zshrc"

# -----------------------------------
# 3. Remove the source directory
# -----------------------------------

if [ -d "$TARGET_DIR" ]; then
    info "Removing source directory $TARGET_DIR..."
    rm -rf "$TARGET_DIR"
else
    info "Source directory $TARGET_DIR not found — nothing to remove."
fi

echo "-------------------------------------------------------"
echo "> UNINSTALL COMPLETE!"
echo ">"
echo "> Conda was left installed. Open a new shell (or run"
echo "> 'source ~/.bashrc') so the removed aliases take effect."
echo "-------------------------------------------------------"
