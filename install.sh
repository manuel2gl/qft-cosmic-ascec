#!/bin/bash
set -e  # Stop immediately if any command fails

#==========================================
# ASCEC ONE-CLICK INSTALLER
#==========================================
# To install ASCEC, simply download and run this script:
#
#   wget https://raw.githubusercontent.com/manuel2gl/qft-cosmic-ascec/main/install.sh
#   bash install.sh
#
# This script will automatically:
#   - Clone/update the repository
#   - Install/configure Miniconda (if needed)
#   - Create appropriate Python environment
#   - Install all dependencies
#   - Set up command shortcuts

#==========================================
# CONFIGURATION
#==========================================
# Set to TRUE to create a separate 'py11' environment with Python 3.11
# Set to FALSE to install into the base conda environment (default)

INSTALL_PY11=TRUE

echo "> Starting ASCEC 'One-Click' Installation..."

# -----------------------------------
# 1. Setup Directory & Download Code
#------------------------------------

TARGET_DIR="$HOME/software/ascec04"
echo "> Setting up directories at $TARGET_DIR..."
mkdir -p "$TARGET_DIR"

if [ -d "$TARGET_DIR/.git" ]; then
    echo "> Repo exists, pulling latest updates..."
    cd "$TARGET_DIR" && git pull
else
    echo "> Cloning repository..."
    git clone https://github.com/manuel2gl/qft-cosmic-ascec.git "$TARGET_DIR"
fi

#-----------------------------------------
# 2. Check for Conda (Install if missing)
#-----------------------------------------

MINICONDA_DIR="$HOME/miniconda3"

if ! command -v conda &> /dev/null; then
    # Conda command not found - check if installation exists
    if [ -d "$MINICONDA_DIR" ]; then
        echo "> Conda installation found at $MINICONDA_DIR. Initializing..."
        eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"
        # Initialize conda in .bashrc if not already done
        "$MINICONDA_DIR/bin/conda" init bash > /dev/null 2>&1
        # If still not working after init, try to update/repair
        if ! command -v conda &> /dev/null; then
            echo "> Updating existing Miniconda installation..."
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
            bash miniconda.sh -b -u -p "$MINICONDA_DIR"
            eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"
            "$MINICONDA_DIR/bin/conda" init bash > /dev/null 2>&1
            rm miniconda.sh
        fi
    else
        echo "> Conda not found. Installing Miniconda..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p "$MINICONDA_DIR"
        eval "$($MINICONDA_DIR/bin/conda shell.bash hook)"
        # Initialize conda in .bashrc
        "$MINICONDA_DIR/bin/conda" init bash > /dev/null 2>&1
        rm miniconda.sh
    fi
else
    echo "> Conda found. Proceeding..."
    # Ensure conda hook is active for this script
    eval "$(conda shell.bash hook)"
    # Ensure conda is initialized in .bashrc
    if ! grep -q "conda initialize" "$HOME/.bashrc"; then
        echo "> Adding conda initialization to .bashrc..."
        conda init bash > /dev/null 2>&1
    fi
fi

#-----------------------------------------
# 3. Check Python Version Compatibility
#-----------------------------------------

# Detect Python version in base environment
PYTHON_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "> Detected Python $PYTHON_VERSION in base environment"

# Check if base Python is compatible
if [ "$INSTALL_PY11" = "FALSE" ]; then
    # Check if Python is too old
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        echo "-------------------------------------------------------"
        echo "> ERROR: Python $PYTHON_VERSION is too old"
        echo "> ASCEC requires Python >= 3.9 or <= 3.11"
        echo "> "
        echo "> SOLUTION: Set INSTALL_PY11=TRUE at the top of this script"
        echo "> This will create a separate environment with Python 3.11"
        echo "-------------------------------------------------------"
        exit 1
    fi
    
    # Check if Python is too new (openbabel incompatible)
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
        echo "-------------------------------------------------------"
        echo "> ERROR: Python $PYTHON_VERSION is incompatible"
        echo "> ASCEC requires Python >= 3.9 or <= 3.11"
        echo "> "
        echo "> SOLUTION: Set INSTALL_PY11=TRUE at the top of this script"
        echo "> This will create a separate environment with Python 3.11"
        echo "-------------------------------------------------------"
        exit 1
    fi
fi

#----------------------------------------------
# 4. Accept Conda Terms of Service
#----------------------------------------------

# Accept Terms of Service for conda channels (required for non-interactive install)
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

# Installs numpy, scipy, etc.
conda install numpy scipy matplotlib scikit-learn -y
# Installs openbabel and cclib from conda-forge
conda install -c conda-forge cclib openbabel -y
# Installs orca-pi parser via pip
pip install orca-pi


#----------------------------------
# 7. Setup Shortcuts (Aliases)
#----------------------------------

echo "> Configuring shortcuts in .bashrc..."
BASHRC="$HOME/.bashrc"

# Add ASCEC shortcuts
if ! grep -q "alias ascec=" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# COSMIC ASCEC aliases" >> "$BASHRC"
    if [ "$INSTALL_PY11" = "TRUE" ]; then
        CONDA_BASE=$(conda info --base)
        PYTHON_BIN="$CONDA_BASE/envs/py11/bin/python"
    else
        PYTHON_BIN="python"
    fi
    echo "alias ascec='$PYTHON_BIN $TARGET_DIR/ascec-v04.py'" >> "$BASHRC"
    echo "alias cosmic='$PYTHON_BIN $TARGET_DIR/cosmic-v01.py'" >> "$BASHRC"
fi

echo "-------------------------------------------------------"
echo "> INSTALLATION COMPLETE!"
echo ">"
echo "> Reload your shell configuration:"
echo "    source ~/.bashrc"
echo ">"
echo "> Then use 'ascec' and 'cosmic' directly — no environment activation needed."
echo "-------------------------------------------------------"
