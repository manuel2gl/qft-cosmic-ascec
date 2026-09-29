<div align="center">

# COSMIC–ASCEC
**Automated Configurational and Conformational Sampling <br> with Topological Screening of Molecular Clusters**

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPL_v3-coral.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Web Interface](https://img.shields.io/badge/Web-Input_Generator-gold?logo=googlechrome&logoColor=white)](https://manuel2gl.github.io/qft-cosmic-ascec/)
[![Documentation](https://img.shields.io/badge/PDF-User_Manual-brightgreen)](./manual.pdf)

**Manuel Gómez • Sara Gómez • Albeiro Restrepo**  
*Química Física Teórica, Universidad de Antioquia, Colombia*

</div>

## What it does

COSMIC–ASCEC performs an automated configurational and conformational search on the potential energy surface of atomic and molecular clusters. From a single input file, built with the web generator, it runs every stage without manual intervention and returns the distinct minima with their relative energies and Boltzmann populations.

1. **ASCEC** (*Annealing Simulado Con Energía Cuántica*) samples the surface by simulated annealing, with energies from xTB or ORCA.
2. **COSMIC** groups the candidates by their physicochemical descriptors, with no atom numbering or superposition, and keeps one motif per family.
3. The motifs are **refined** at a higher level of theory, with frequencies that confirm true minima, and screened again.

Every run is one of three modes: **preliminary** (a semiempirical map in minutes), **rigorous** (verified minima with Boltzmann populations) or **ultimate** (adds high level single point energies). COSMIC also works on its own, on any set of structures.

### Key features

| Feature | Description |
| :--- | :--- |
| **Automated workflow** | Annealing, preoptimization, clustering and refinement from a single command, resumable after any interruption. |
| **Failure recovery** | Relaunches crashed jobs and displaces structures with imaginary frequencies until they reach true minima. |
| **Similarity screening** | Hierarchical clustering on physicochemical feature vectors, independent of atom numbering, with optional RMSD refinement. |
| **Conformational sampling** | Internal dihedral rotations sampled together with the arrangement of the molecules. |
| **QM backends** | xTB 6.7+ and ORCA 5.0.x or 6.1+. |
| **Web input generator** | Fetches molecules from PubChem, previews the box in 3D and builds the whole protocol. |

## Installation

**Prerequisites:** [git](https://git-scm.com/downloads) and an internet connection. The installer creates a Python 3.11 environment (`py11`) with every dependency, including xTB, and needs no administrator rights.

**Linux**, with one command:

```bash
cd "$HOME" && wget \
  https://raw.githubusercontent.com/manuel2gl/qft-cosmic-ascec/main/install.sh \
  && bash install.sh && rm -f install.sh && source ~/.bashrc
```

**Windows:**

1. Download [`win_install.bat`](https://raw.githubusercontent.com/manuel2gl/qft-cosmic-ascec/main/win_install.bat).
2. Double click it. If SmartScreen warns that it was downloaded from the internet, choose **More info → Run anyway**.
3. Open a new terminal so the PATH change takes effect.

Only preliminary runs have been tested on Windows, and `ascec status`, detaching and the `after` queue are Linux only.

> [!NOTE]
> Python 3.10 or newer is required. The installers create a 3.11 environment, so this only matters if you manage your own.

Check the installation with `ascec --version`. Installing on a cluster, installing by hand, uninstalling and installing ORCA (needed for rigorous and ultimate runs) are covered in Section 2 of the [manual](./manual.pdf).

> [!WARNING]
> ORCA 6.0 is not supported. Use ORCA 5.0.x or 6.1+.

## Quick start

Build an input in the [web input generator](https://manuel2gl.github.io/qft-cosmic-ascec/), or open it from the terminal with `ascec input`. Then run it:

```bash
ascec system.asc
```

The protocol embedded in the file runs every stage in order:
`Annealing` ➔ `Preoptimization` ➔ `COSMIC` ➔ `Refinement` ➔ `COSMIC` ➔ `Boltzmann populations`.
A long run can be detached with `Ctrl+D`, followed with `ascec status`, and resumed by running the same command again.

To try a ready made input first, copy the glycolaldehyde and water example and run it (under a minute):

```bash
cp -r ~/software/ascec04/examples/glyw2/preliminary glyw2 && cd glyw2 && ascec glyw2.asc
```

Each stage can also be run on its own, for example to check the box and launch three annealing replicas at 20% packing:

```bash
ascec system.asc box
ascec system.asc r3 --box20
./launcher_ascec.sh
```

## Output

* `final_ensemble.xyz`: the unique minima, ranked by population; the first frame is the putative global minimum.
* `boltzmann_distribution.txt`: Gibbs energies and Boltzmann populations.
* `tvse_*.png`: energy evolution of each annealing replica.
* `result_*.xyz`: every accepted configuration, ready for Avogadro, GaussView or IQmol.
* Dendrograms and threshold diagnostics for every COSMIC pass.

## Documentation

* [`manual.pdf`](./manual.pdf): the user manual. Part I gets a new user from installation to a finished run; Part II covers the theory, the input format and advanced use.
* [`examples/`](./examples/): the inputs of every system and mode used in the manual.
* [`docs/`](./docs/): earlier ASCEC studies of water, methanol, formic acid, gold and other clusters.

## Citation and license

If you use COSMIC–ASCEC in published work, please cite:

* M. Gómez, S. Gómez, A. Restrepo, *[title, journal and DOI to be added; manuscript under consideration]*.
* Reference data: Zenodo, DOI [10.5281/zenodo.20723683](https://doi.org/10.5281/zenodo.20723683).

COSMIC–ASCEC is free software under the GNU General Public License v3; see [`license`](./license).
