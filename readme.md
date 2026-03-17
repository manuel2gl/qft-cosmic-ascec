<div align="center">

# ASCEC & Similarity
**Automated Configurational Sampling and Topological Screening of Molecular Clusters**

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPL_v3-coral.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Web Interface](https://img.shields.io/badge/Web-Input_Generator-gold?logo=googlechrome&logoColor=white)](https://manuel2gl.github.io/qft-ascec-similarity/)
[![Documentation](https://img.shields.io/badge/📖-User_Manual-brightgreen)](#-documentation)

*Bridging the gap between stochastic chaos and ordered chemical insight.*

<br>

**Manuel Gómez • Sara Gómez • Albeiro Restrepo**  
*Química Física Teórica, Instituto de Química*  
**Universidad de Antioquia, Colombia**

</div>

---

## What is ASCEC?

**ASCEC** (*Annealing Simulado Con Energía Cuántica*) is an intelligent computational orchestrator designed to explore complex potential energy surfaces (PES). By pairing robust stochastic sampling (Simulated Annealing) with the **Similarity** topological clustering module, it automates the discovery of low-energy molecular conformations.

It eliminates the tedious manual processing of thousands of configurations by automatically filtering redundancies, correcting imaginary frequencies, and refining unique minima using high-level quantum mechanical (QM) evaluations.

### Key Features

| Feature | Description |
| :--- | :--- |
| 🤖 **Fully Automated Workflows** | Execute multi-stage pipelines (Annealing → Pre-optimization → Clustering → Refinement) with a single command. |
| 🧠 **Intelligent Recovery** | Automatically handles calculation crashes and perturbs structures to eliminate imaginary frequencies (transition states). |
| 📊 **Hierarchical Clustering** | Identifies representative structures using continuous physicochemical feature vectors, with optional RMSD secondary refinement. |
| ⚡ **Multi-Backend QM Support** | Seamlessly interfaces with **ORCA** (v5.0.x and v6.1+) and **Gaussian** (09/16). |
| 🌐 **Web Interface** | Interactive browser tool to fetch molecules via PubChem, visualize the simulation box, and build input files effortlessly. |

---

## 📖 Documentation

For a comprehensive guide covering the theoretical background, detailed parameter explanations, calculation setups, and advanced tutorials (e.g., Water Hexamer and Formic Acid Dimer workflows), please consult the official User Manual:

<div align="center">

[![Download Manual](https://img.shields.io/badge/PDF-Download_ASCEC_User_Manual-red?style=for-the-badge&logo=adobe-acrobat-reader)](./manual.pdf)

</div>

> [!NOTE]  
> We highly recommend reviewing the **Optimization Strategy** and **Similarity Clustering** sections in the manual to understand how to correctly select thresholds ($\tau$) and handle skipped/critical geometries.

---

## ⚙️ Installation

ASCEC requires **Python 3.9+** (3.11 recommended) and utilizes an external electronic structure package (ORCA or Gaussian) as a backend. 

> [!WARNING]  
> ORCA 6.0 is **not** supported due to parser limitations. Please use ORCA v5.0.x or upgrade to **v6.1+**.

### Option 1: Automatic "One-Click" Installation (Recommended)
We provide a unified shell script that automates the entire setup. It will install Miniconda (if missing), set up a dedicated Python 3.11 environment (`py11`), install all dependencies (`cclib`, `orca-pi`, `openbabel`), and configure your terminal aliases.

#### 1. Download the installation script
```bash
wget https://raw.githubusercontent.com/manuel2gl/qft-ascec-similarity/main/install.sh
```
#### 2. Run the script
```
bash install.sh
```
#### 3. Reload your terminal configuration
```bash
source ~/.bashrc
```

### Option 2: Step-by-Step Conda Installation
If you prefer to manage your environments manually, you can set up ASCEC using Conda:

**1. Clone the repository:**
```bash
mkdir -p ~/software/ascec04
git clone https://github.com/manuel2gl/qft-ascec-similarity.git ~/software/ascec04/
```

**2. Create and activate a clean environment:**
```bash
conda create -n py11 python=3.11 -y
conda activate py11
```

**3. Install dependencies:**
```bash
conda install numpy scipy matplotlib scikit-learn -y
conda install -c conda-forge cclib openbabel -y
pip install orca-pi
```

**4. Set up aliases:**
Add the following lines to your `~/.bashrc` (or `~/.zshrc`) to make the commands globally available:
```bash
alias ascec='python $HOME/software/ascec04/ascec-v04.py'
alias simil='python $HOME/software/ascec04/similarity-v01.py'
```
Run `source ~/.bashrc` to apply the changes.

---

## Quick Start & Usage

### 1. Generate Input via Web Interface
Use the ASCEC Web Generator with built-in PubChem integration to instantly get 3D coordinates and visualize your simulation box.

**[ASCEC Web Input Generator](https://manuel2gl.github.io/qft-ascec-similarity/)**

Alternatively, launch it directly from your terminal:
```bash
ascec input
```

### 2. Standalone Annealing Simulation
Once your input file (e.g., `system.asc`) is generated, you can validate the simulation box and launch the annealing process:
```bash
# Analyze simulation box requirements
ascec system.asc box

# Run in triplicate (r3) using a 10% effective packing box
ascec system.asc r3 --box10

# Execute the generated launcher
./launcher_ascec.sh
```

### 3. Fully Automated Workflow Protocol
ASCEC truly shines when automating the tedious optimization and clustering cycles. Define a multi-stage workflow in your input file and launch it with a single command:

```bash
ascec system.asc
```
*The workflow will autonomously manage:*<br>
`Annealing` ➔ `Pre-optimization (e.g., GFN2-xTB)` ➔ `Topological Clustering` ➔ `High-level DFT Refinement` ➔ `Final Boltzmann Analysis`.

---

## Output & Visual Analytics

ASCEC automatically organizes your data and generates publication-ready analytics:
*   📉 **`tvse_*.dat / .png`**: Energy evolution profiles across Monte Carlo steps.
*   💧 **`result_*.xyz`**: Complete trajectory files ready for visualization in Avogadro, GaussView, or IQmol.
*   🌳 **Dendrograms**: Beautiful hierarchical tree plots (`.png`) visually detailing the clustering distances of distinct structural families.
*   🧮 **Boltzmann Distribution**: A concise `.txt` summary ranking unique configurations by their Gibbs Free Energy populations.

---

## 📄 License & Citation

ASCEC & Similarity is free software distributed under the **GNU General Public License (GPL) version 3**. See the `license` file for more details.

If you use ASCEC or Similarity in your research, please acknowledge the software and the developers. The theoretical implementation of the Modified Metropolis test and topological clustering is based on extensive prior structural studies. Please refer to the User Manual's bibliography for specific literature pertaining to the utilized algorithms.