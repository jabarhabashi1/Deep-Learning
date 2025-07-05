# HyperMinerDL : Deep‑Learning Pipeline for Hyperspectral Mineral Mapping

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**HyperMinerDL** is an end‑to‑end, research‑grade workflow designed to derive critical mineralogical insights from hyperspectral PRISMA satellite imagery over extreme environments such as the Dry Valleys (South Victoria Land, Antarctica). It bundles three independent yet interoperable Python scripts:

| Script                       | Purpose                                                                                           | Core Algorithm                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `Augmentation_for_shaire.py` | Generates physically plausible offsets of laboratory spectral libraries to enlarge training data. | Offset‑based spectral augmentation                                  |
| `AVCA_for_shairing.py`       | Extracts scene endmembers via **Adaptive/Vertex Component Analysis (AVCA)**.                      | Python translation and optimisation of Nascimento & Dias (2005) VCA |
| `CNN_for_shaire.py`          | Performs 3‑D convolutional neural network classification and abundance‑mapping of PRISMA imagery. | 3‑D CNN + early stopping + class‑wise abundance mapping             |

---

## Table of Contents

1. [Project Highlights](#project-highlights)
2. [Getting Started](#getting-started)
3. [Data Preparation](#data-preparation)
4. [Usage](#usage)

   * [1  Spectral Augmentation](#1-spectral-augmentation)
   * [2  AVCA Endmember Extraction](#2-avca-endmember-extraction)
   * [3  3‑D CNN Classification](#3-3-d-cnn-classification)
5. [Outputs](#outputs)
6. [Reproducing the Paper Results](#reproducing-the-paper-results)
7. [Citing This Work](#citing-this-work)
8. [License](#license)
9. [Contributing](#contributing)
10. [Contact](#contact)
11. [Acknowledgements](#acknowledgements)

---

## Project Highlights

* **Research‑backed**: Code accompanies the manuscript *“Revealing critical mineralogical insights in extreme environments using deep learning technique on hyperspectral PRISMA satellite imagery”* (ISPRS J. Photogrammetry & Remote Sensing, 2025).
* **Modular**: Each script can be executed standalone or chained for a full workflow.
* **Reproducible**: Clear environment specification, seed control, and line‑level comments indicating where to plug in your own data.
* **Extensible**: Replace PRISMA with other hyperspectral sensors or swap the CNN architecture with minimal refactoring.

---

## Getting Started

### Prerequisites

```bash
Python >= 3.8  # 3.11 tested
pip >= 22.0
```

**Python dependencies** (install automatically via `requirements.txt`):

| Package                 | Purpose                          |
| ----------------------- | -------------------------------- |
| `numpy`                 | numerical operations             |
| `rasterio`              | raster I/O                       |
| `spectral`              | ENVI / spectral library handling |
| `tensorflow` ≥ 2.15     | deep learning backend            |
| `scikit‑learn`          | train‑test split, metrics        |
| `matplotlib`, `seaborn` | visualisation                    |
| `warnings`, `os`, `re`  | utilities                        |

```bash
# 1️⃣ Clone the repo
git clone https://github.com/<your_username>/HyperMinerDL.git
cd HyperMinerDL

# 2️⃣ Create a fresh environment (conda example)
conda create -n hyperminer python=3.11
conda activate hyperminer

# 3️⃣ Install dependencies
pip install -r requirements.txt
```

> **GPU acceleration**: TensorFlow will automatically detect CUDA‑enabled GPUs if the appropriate CUDA & cuDNN libraries are on your `PATH`. Otherwise it falls back to CPU.

---

## Data Preparation

You will need:

| Data Item                        | Format                 | Example                | Used‑by           |
| -------------------------------- | ---------------------- | ---------------------- | ----------------- |
| **PRISMA hyperspectral cube**    | ENVI (`.hdr` + `.dat`) | `DryValley_PRISMA.dat` | AVCA, CNN         |
| **Mask raster** (optional)       | ENVI/GeoTIFF           | `IceRock_mask.dat`     | AVCA, CNN         |
| **HDR metadata**                 | `.hdr`                 | `DryValley_PRISMA.hdr` | AVCA, CNN         |
| **Spectral library** per mineral | `.sli` + `.hdr`        | `Chlorite.sli`         | Augmentation, CNN |

> 📌 **Placeholders** in each script (`"import RS data path"`, `"import mask path"`, etc.) must be updated to point to your own files. Refer to the *Importing Data* comments around the indicated line numbers (see below).

---

## Usage

Run each stage from the project root or integrate them into your own pipeline.

### 1. Spectral Augmentation

Produces multiple offset versions of a spectral library, mitigating limited reference spectra.

```bash
python Augmentation_for_shaire.py \
    --input r"C:\data\USGS_ASCL\Veg.txt" \
    --output Vegetation_augmented.txt \
    --initial_offset 0 \
    --increment 0.01 \
    --iterations 30
```

*Result*: `Vegetation_augmented.txt` (first column = wavelength; subsequent columns = augmented reflectances).

### 2. AVCA Endmember Extraction

Identifies the purest pixels (endmembers) in the scene.

```bash
python AVCA_for_shairing.py \
    --raster   r"C:\data\DryValley_PRISMA.dat" \
    --mask     r"C:\data\IceRock_mask.dat" \
    --hdr      r"C:\data\DryValley_PRISMA.hdr" \
    --num_endmembers 10
```

*Result*: `AVCA_results.txt` in the same directory, listing wavelength‑wise reflectances for each endmember.

### 3. 3‑D CNN Classification

Learns from the (augmented) spectral library to produce class and abundance maps.

```bash
python CNN_for_shaire.py \
    --rs_data     r"C:\data\DryValley_PRISMA.dat" \
    --rs_hdr      r"C:\data\DryValley_PRISMA.hdr" \
    --mask        r"C:\data\IceRock_mask.dat" \
    --spectra_dir r"C:\data\Spectra_Libraries" \
    --epochs 300 --batch 64
```

*Result*: `4 classified.hdr/dat` + abundance maps and confusion‑matrix plots displayed during runtime.

> **Early stopping** triggers automatically once both training & validation accuracy exceed 0.993 **and** losses drop below 0.11.

---

## Outputs

```text
outputs/
├─ Vegetation_augmented.txt           # 1. Spectral augmentation
├─ AVCA_results.txt                   # 2. Endmember extraction
└─ 4 classified.hdr / .dat            # 3. CNN abundance maps
```

Additional artefacts (accuracy/loss graphs, confusion matrix, per‑class abundance PNGs) are saved in the run directory.

---

## Reproducing the Paper Results

1. Download the PRISMA scene (`PRISMA_L1_HI_20200129T235623_20200130T000131.dat`) and its HDR.
2. Acquire the USGS ASCL spectral library for relevant minerals.
3. Run **Spectral Augmentation** → **AVCA** → **CNN** exactly as above.
4. Compare the generated abundance maps with Fig. 10 in the paper.

---

## Citing This Work

If you use **HyperMinerDL** or parts of its code, please cite the companion article:

> Habashi, J. *etal.* (2025). **Revealing critical mineralogical insights in extreme environments using deep learning technique on hyperspectral PRISMA satellite imagery: Dry Valleys, South Victoria Land, Antarctica.** *ISPRS Journal of Photogrammetry & Remote Sensing*. DOI: *TBD*

BibTeX:

```bibtex
@article{habashi2025hyperminer,
  author  = {Habashi, Jabar and Beiranvand Pour, Amin and Muslim, Aidy M. and Moradi Afrapoli, Ali and Hong, Jong Kuk and Park, Yongcheol and Almasi, Alireza and Crispini, Laura and Hashim, Mazlan and Bagheri, Milad},
  title   = {Revealing critical mineralogical insights in extreme environments using deep learning technique on hyperspectral PRISMA satellite imagery: Dry Valleys, South Victoria Land, Antarctica},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year    = 2025,
  doi     = {<to‑be‑assigned>}
}
```

---

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details. **Any reuse must include a citation of the manuscript above.**

---

## Contributing

Pull requests are welcome! If you plan a substantial contribution (new model, sensor support, etc.) please open an issue first to discuss the design.

1. Fork the project and create your feature branch (`git checkout -b feature/new‑sensor`).
2. Commit your changes with clear messages.
3. Ensure existing tests pass (`pytest`).
4. Open a pull request and describe your enhancement.

---

## Contact

For questions relating to the code or paper, please open a GitHub issue or email **jabar.habashi \[at] example.com**.

---

## Acknowledgements

* Original MATLAB VCA algorithm by José Nascimento & José Bioucas Dias (2005).
* USGS ASCL spectral library.
* TensorFlow, Scikit‑Learn, and the open‑source GIS community.

---

<sub>README generated 🗓 05 Jul 2025 – 🇫🇷 UTC+02:00</sub>
