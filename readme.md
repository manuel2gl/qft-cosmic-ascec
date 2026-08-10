<div align="center">

# COSMIC-ASCEC
**Automated Configurational Sampling and Topological Screening of Molecular Clusters**

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPL_v3-coral.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Web Interface](https://img.shields.io/badge/Web-Input_Generator-gold?logo=googlechrome&logoColor=white)](https://manuel2gl.github.io/qft-cosmic-ascec/)
[![Documentation](https://img.shields.io/badge/📖-User_Manual-brightgreen)](#-documentation)

*Bridging the gap between stochastic chaos and ordered chemical insight.*

<br>

**Manuel Gómez • Sara Gómez • Albeiro Restrepo**  
*Química Física Teórica, Instituto de Química*  
**Universidad de Antioquia, Colombia**

</div>

## What is COSMIC-ASCEC?

**COSMIC-ASCEC** is a Python tool that automatically samples the configurational potential energy surface of molecular clusters and screens their topological features. The name joins its two engines, the ASCEC annealing search (*Annealing Simulado Con Energía Cuántica*) and the COSMIC topological clustering module.

Acting as an intelligent computational orchestrator, COSMIC-ASCEC pairs robust stochastic sampling (simulated annealing) with topological clustering to automate the discovery of low energy molecular conformations. It removes the tedious manual processing of thousands of configurations by automatically filtering redundancies, correcting imaginary frequencies, and refining unique minima with high level quantum mechanical (QM) evaluations.

### Key Features

| Feature | Description |
| :--- | :--- |
| 🤖 **Fully Automated Workflows** | Execute pipelines (annealing, then preoptimization, then clustering, then refinement) with a single command. |
| 🧠 **Intelligent Recovery** | Automatically handles calculation crashes and perturbs structures to remove imaginary frequencies (transition states). |
| 📊 **Hierarchical Clustering** | Identifies representative structures using continuous physicochemical feature vectors, with optional RMSD refinement. |
| ⚡ **Multiple QM Backends** | Interfaces seamlessly with **ORCA** (v5.0.x and v6.1+) and **xTB** (v6.7.1). |
| 🌐 **Web Interface** | Interactive browser tool to fetch molecules via PubChem, visualize the simulation box, and build input files effortlessly. |

## 📖 Documentation

For a comprehensive guide covering the theoretical background, detailed parameter explanations, calculation setups, and advanced tutorials (for example the water hexamer and formic acid dimer workflows), please consult the official User Manual.

Worked examples for the systems discussed in the manual live in [`docs/`](./docs/) as PDFs (water clusters, formic acid dimer, methanol tetramer, gold clusters, and more).

<div align="center">

[![Download Manual](https://img.shields.io/badge/PDF-Download_COSMIC--ASCEC_User_Manual-red?style=for-the-badge&logo=adobe-acrobat-reader)](./manual.pdf)

</div>

> [!NOTE]  
> We highly recommend reviewing the **Optimization Strategy** and **COSMIC Clustering** sections in the manual to understand how to correctly select thresholds ($\tau$) and handle skipped or critical geometries.

## ⚙️ Installation

COSMIC-ASCEC requires **Python 3.10+** (3.11 recommended) and uses an external electronic structure package (ORCA or xTB) as a backend.

> [!IMPORTANT]  
> **Python 3.10 is a hard floor.** The `.asc` parser uses a `match` statement, which is a *syntax* error on 3.9 — a 3.9 install appears to succeed and then fails on the first run. The installers create a 3.11 environment, so this only matters if you manage your own.

> [!WARNING]  
> ORCA 6.0 is **not** supported due to parser limitations. Please use ORCA v5.0.x, or upgrade to **v6.1+**.

### Prerequisites

| Requirement | Notes |
| :--- | :--- |
| **git** | Installed automatically if missing — from conda-forge on Linux, from winget or conda-forge on Windows. If neither works the installer falls back to downloading a source archive, so a git-free install is still possible. Manual download: [git-scm.com](https://git-scm.com/download/win) |
| **Network** | Needed for the initial download; nothing phones home afterwards |
| **Admin rights** | Not required on either platform |

### Option 1: Automatic "One Click" Installation (Recommended)

The installer bootstraps Miniconda (if missing), creates a dedicated Python 3.11 environment (`py11`), installs every dependency, and wires up the `ascec` / `cosmic` commands. They point directly at the environment's Python binary, so **no manual `conda activate` is needed**.

#### 🐧 Linux

**1. Download the installation script**
```bash
wget https://raw.githubusercontent.com/manuel2gl/qft-cosmic-ascec/main/install.sh
```
**2. Run the script**
```bash
bash install.sh
```
**3. Reload your terminal configuration**
```bash
source ~/.bashrc
```

Aliases are written into a fenced block in `~/.bashrc` (and `~/.zshrc` if present):

```bash
# >>> COSMIC ASCEC >>>
...
# <<< COSMIC ASCEC <<<
```

Only that block is rewritten on reinstall and removed on uninstall, so your own aliases are never touched. Re-running `install.sh` on a checkout with uncommitted edits **skips** the `git pull` and tells you why, rather than discarding your work.

#### 🪟 Windows

**1.** Download [`win_install.bat`](https://raw.githubusercontent.com/manuel2gl/qft-cosmic-ascec/main/win_install.bat)
**2.** Double-click it in File Explorer
**3.** Open a **new** terminal so the PATH change takes effect

It installs into `%USERPROFILE%\software\ascec04`, writes `ascec.cmd` / `cosmic.cmd` into `%USERPROFILE%\bin`, and adds that directory to your user PATH.

> [!NOTE]  
> Because the file was downloaded from the internet, Windows marks it and SmartScreen may show *"Windows protected your PC"*. Choose **More info → Run anyway**. The installer is plain, readable batch — open it in Notepad first if you want to see exactly what it does.

To uninstall: `bash uninstall.sh` on Linux, or double-click `win_uninstall.bat` on Windows. Both remove the `py11` environment, the commands and the source directory, and both leave conda itself installed.

> [!NOTE]  
> **Windows support covers running calculations.** Annealing, preoptimization, clustering and refinement all work. `ascec status`, Ctrl+D detach and the `after <PID>` background queue are **Linux-only** — they depend on process groups, signals and `/proc`. Running `ascec status` on Windows prints a message saying so rather than showing a broken screen.

### Option 2: Step by Step Conda Installation
If you prefer to manage your environments manually, you can set up COSMIC-ASCEC using Conda.

**1. Clone the repository:**
```bash
mkdir -p ~/software/ascec04
git clone https://github.com/manuel2gl/qft-cosmic-ascec.git ~/software/ascec04/
```

**2. Create and activate a clean environment:**
```bash
conda create -n py11 python=3.11 -y
conda activate py11
```

**3. Install dependencies:**
```bash
conda install -c conda-forge --override-channels -y \
    numpy scipy matplotlib cclib openbabel "xtb>=6.7"
pip install orca-pi
```

Everything comes from conda-forge deliberately — mixing it with the `defaults` channel can produce ABI mismatches between packages that share libraries. The `xtb>=6.7` floor matters too: conda-forge's 6.5.0 build crashes mid-optimization and silently yields zero motifs.

The dependency list also lives in [`environment.yml`](./environment.yml) and [`pyproject.toml`](./pyproject.toml), so steps 2–3 collapse to:

```bash
conda env create -f environment.yml
conda activate py11
```

**4. Set up aliases:**
Add the following lines to your `~/.bashrc` (or `~/.zshrc`) to make the commands globally available. Use the full path to the environment's Python binary so no activation is needed:
```bash
# Replace <conda_base> with the output of: conda info --base
alias ascec='<conda_base>/envs/py11/bin/python $HOME/software/ascec04/ascec-v04.py'
alias cosmic='<conda_base>/envs/py11/bin/python $HOME/software/ascec04/cosmic-v01.py'
```
Run `source ~/.bashrc` to apply the changes.

## Quick Start and Usage

### 1. Generate Input via Web Interface
Use the COSMIC-ASCEC Web Generator with built in PubChem integration to instantly get 3D coordinates and visualize your simulation box.

**[COSMIC-ASCEC Web Input Generator](https://manuel2gl.github.io/qft-cosmic-ascec/)**

Alternatively, launch it directly from your terminal:
```bash
ascec input
```

### 2. Standalone Annealing Simulation
Once your input file (for example `system.asc`) is generated, you can validate the simulation box and launch the annealing process:
```bash
# Analyze simulation box requirements
ascec system.asc box

# Run in triplicate (r3) using a 10% effective packing box
ascec system.asc r3 --box10

# Execute the generated launcher
./launcher_ascec.sh
```

### 3. Fully Automated Workflow Protocol
COSMIC-ASCEC truly shines when automating the tedious optimization and clustering cycles. Define a workflow in your input file and launch it with a single command:

```bash
ascec system.asc
```
*The workflow will autonomously manage:*<br>
`Annealing` ➔ `Preoptimization (for example GFN2-xTB)` ➔ `Topological Clustering` ➔ `High level DFT Refinement` ➔ `Final Boltzmann Analysis`.

## Output and Visual Analytics

COSMIC-ASCEC automatically organizes your data and generates publication ready analytics:
*   📉 **`tvse_*.dat / .png`**: Energy evolution profiles across Monte Carlo steps.
*   💧 **`result_*.xyz`**: Complete trajectory files ready for visualization in Avogadro, GaussView, or IQmol.
*   🌳 **Dendrograms**: Hierarchical tree plots (`.png`) that visually detail the clustering distances of distinct structural families.
*   🧮 **Boltzmann Distribution**: A concise `.txt` summary ranking unique configurations by their Gibbs free energy populations.

## 📄 License and Citation

COSMIC-ASCEC is free software distributed under the **GNU General Public License (GPL) version 3**. See the `license` file for more details.

If you use COSMIC-ASCEC in your research, please acknowledge the software and the developers. The theoretical implementation of the modified Metropolis test and topological clustering is based on extensive prior structural studies. Please refer to the User Manual's bibliography for the specific literature on the algorithms used.
