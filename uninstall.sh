#!/bin/bash
set -e  # Stop immediately if any command fails

#==========================================
# COSMIC ASCEC v04 UNINSTALLER (Linux)
#==========================================
# Reverses what install.sh set up, WITHOUT touching conda itself:
#   - Removes the 'py11' conda environment (leaves base conda alone)
#   - Removes the cloned/copied source directory ($HOME/software/ascec04)
#   - Removes the `ascec` and `cosmic` aliases from ~/.bashrc
#
# Conda (Miniconda/Anaconda) is intentionally left installed, since the user
# may rely on it for other work. Run with --yes to skip the confirmation prompt.

ENV_NAME="py11"
TARGET_DIR="$HOME/software/ascec04"
BASHRC="$HOME/.bashrc"

ASSUME_YES=FALSE
for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=TRUE ;;
    esac
done

echo "> COSMIC ASCEC v04 uninstaller"
echo ">"
echo "> This will:"
echo ">   - Remove the conda environment '$ENV_NAME' (conda itself is kept)"
echo ">   - Remove the source directory $TARGET_DIR"
echo ">   - Remove the 'ascec'/'cosmic' aliases from $BASHRC"
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
# current shell hasn't been `conda init`-ed.
if ! command -v conda &> /dev/null; then
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/conda" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
        if [ -x "$candidate/bin/conda" ]; then
            eval "$("$candidate/bin/conda" shell.bash hook)"
            break
        fi
    done
fi

if command -v conda &> /dev/null; then
    if conda env list | grep -q "^$ENV_NAME "; then
        echo "> Removing conda environment '$ENV_NAME'..."
        # Deactivate first if we happen to be inside it, or removal fails.
        conda deactivate 2>/dev/null || true
        conda env remove -n "$ENV_NAME" -y
    else
        echo "> Conda environment '$ENV_NAME' not found — nothing to remove."
    fi
else
    echo "> WARNING: conda not found on PATH or in common locations."
    echo "  Skipping env removal. If '$ENV_NAME' exists, remove it manually:"
    echo "    conda env remove -n $ENV_NAME"
fi

# -----------------------------------
# 2. Remove the aliases from .bashrc
# -----------------------------------

if [ -f "$BASHRC" ]; then
    echo "> Removing aliases from $BASHRC..."
    sed -i '/# COSMIC ASCEC aliases/d' "$BASHRC"
    sed -i '/alias ascec=/d' "$BASHRC"
    sed -i '/alias cosmic=/d' "$BASHRC"
fi

# -----------------------------------
# 3. Remove the source directory
# -----------------------------------

if [ -d "$TARGET_DIR" ]; then
    echo "> Removing source directory $TARGET_DIR..."
    rm -rf "$TARGET_DIR"
else
    echo "> Source directory $TARGET_DIR not found — nothing to remove."
fi

echo "-------------------------------------------------------"
echo "> UNINSTALL COMPLETE!"
echo ">"
echo "> Conda was left installed. Open a new shell (or run"
echo "> 'source ~/.bashrc') so the removed aliases take effect."
echo "-------------------------------------------------------"
