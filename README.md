# Hyperspectral Mineral Mapping Toolbox

### (Code Companion to the Manuscript:

*Revealing Critical Mineralogical Insights in Extreme Environments Using Deep‑Learning on PRISMA Hyperspectral Imagery – Dry Valleys, South Victoria Land, Antarctica*)

> **Authors:** Jabar Habashi · Amin Beiranvand Pour · Aidy M Muslim · Ali Moradi Afrapoli · Jong Kuk Hong · Yongcheol Park · Alireza Almasi · Laura Crispini · Mazlan Hashim · Milad Bagheri
> **Journal:** *ISPRS Journal of Photogrammetry and Remote Sensing*
> **Status / DOI:** *(in press – DOI forthcoming)*

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Repository Layout](#repository-layout)
3. [Quick Start](#quick-start)

   1. [Environment](#environment)
   2. [Installation](#installation)
4. [Workflow](#workflow)

   1. [1️⃣ Spectral‑Library Augmentation](#1️⃣-spectral‑library-augmentation)
   2. [2️⃣ Adaptive VCA Endmember Extraction](#2️⃣-adaptive-vca-endmember-extraction)
   3. [3️⃣ 3‑D CNN Classification & Abundance Mapping](#3️⃣-3‑d-cnn-classification--abundance-mapping)
5. [Expected Inputs & Directory Structure](#expected-inputs--directory-structure)
6. [Outputs](#outputs)
7. [Reproducibility Checklist](#reproducibility-checklist)
8. [Citation & License](#citation--license)
9. [Contact & Support](#contact--support)

---

## Overview

This repository hosts three standalone yet interoperable Python scripts that together comprise the complete processing chain used in the above‑cited article to **augment spectral libraries, extract endmembers and unmix hyperspectral PRISMA imagery with a deep 3‑D Convolutional Neural Network (CNN)**.  Each module can be executed independently, but maximum performance is obtained when they are run sequentially:

| Step | Script                       | Purpose                                                                                                                                                                               |
| ---- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `Augmentation_for_shaire.py` | Generates physics‑inspired spectral perturbations (offset‑based) to enlarge sparse laboratory spectral libraries and improve model generalisation.                                    |
| 2    | `AVCA_for_shairing.py`       | Python translation and extension of the Vertex Component Analysis (VCA) algorithm with adaptive SNR‑aware projection ("AVCA"). Extracts endmember signatures from PRISMA image cubes. |
| 3    | `CNN_for_shaire.py`          | End‑to‑end 3‑D CNN classifier that fuses the augmented libraries, AVCA endmembers and masked image data to produce mineral class maps and per‑class abundance fractions.              |

All three scripts are released under the **MIT License** with the requirement that users cite the accompanying manuscript if any part of the code is employed in academic or commercial work.

> **Important:** The scripts contain **placeholder paths** (marked by comments such as `# import RS data path`) that must be updated to match your local directory structure before execution.

---

## Repository Layout

```
📂 hyperspectral‑mineral‑mapping/
├─ Augmentation_for_shaire.py          # Spectral offset augmentation
├─ AVCA_for_shairing.py               # Adaptive Vertex Component Analysis
├─ CNN_for_shaire.py                  # 3‑D CNN classifier & mapper
├─ examples/
│   ├─ spectra/                       # Sample spectral libraries (.sli / .hdr)
│   ├─ prisma_scene/                  # Sample PRISMA cube (.dat / .hdr)
│   └─ mask/                          # Example binary mask
├─ docs/                              # Supplementary figures & slides
└─ README.md                          # You are here ✅
```

Feel free to reorganise as long as the internal path variables in the scripts are updated accordingly.

---

## Quick Start

### Environment

```bash
# Clone the repo
$ git clone https://github.com/<YourUser>/hyperspectral‑mineral‑mapping.git
$ cd hyperspectral‑mineral‑mapping

# (Recommended) Create a dedicated environment
$ conda create -n hypermap python=3.10
$ conda activate hypermap

# Install core dependencies
$ pip install -r requirements.txt
```

**Minimum tested package versions** (adjust as needed):

* numpy ≥ 1.25
* rasterio ≥ 1.3
* spectral ≥ 0.23
* tensorflow ≥ 2.15 (with GPU support preferred)
* scikit‑learn ≥ 1.4
* seaborn ≥ 0.13
* matplotlib ≥ 3.9

> **Tip:** GPU acceleration (CUDA 11+) reduces CNN training time by \~10×.

### Installation

No additional compilation is required; the scripts are pure Python. After installing the dependencies, you are ready to run the pipeline.

---

## Workflow

The following diagram summarises the recommended execution order:

```
           ┌──────────────────┐        ┌─────────────────┐        ┌────────────────────┐
           │  Spectral SLI    │        │  PRISMA Cube    │        │   Binary Mask      │
           └────────┬─────────┘        └────────┬────────┘        └────────┬──────────┘
                    │                          │                          │
                    ▼                          │                          │
   Augmentation_for_shaire.py                  │                          │
         (augmented *.txt)                     │                          │
                    │                          │                          │
                    │                          ▼                          │
                    │              AVCA_for_shairing.py                   │
                    │              (endmember matrix)                     │
                    │                          │                          │
                    └──────────────┬──────────┴──────────┬───────────────┘
                                   ▼                     ▼
                         CNN_for_shaire.py  –––▶  Classified Map &
                                                Abundance Cubes
```

### 1️⃣ Spectral‑Library Augmentation

```bash
$ python Augmentation_for_shaire.py \
      --input r"C:\path\to\ASCL_library.sli" \
      --output Vegetation.txt \
      --iterations 30 --offset 0.01
```

* **Input:** ENVI‑classic spectral library (`*.sli`) of mineral reflectances.
* **Output:** `Vegetation.txt` – n × (m·iter) matrix where *n* is the number of wavelengths and *m* is the original number of spectra.

Modify `initial_offset`, `increment` and `num_iterations` in‑code or expose them via CLI flags to tailor augmentation strength.

### 2️⃣ Adaptive VCA Endmember Extraction

```bash
$ python AVCA_for_shairing.py \
      --raster r"C:\path\prisma\scene.dat" \
      --mask   r"C:\path\mask.dat" \
      --hdr    r"C:\path\prisma\scene.hdr" \
      --R 15
```

* **Input:** PRISMA hyperspectral cube (`*.dat`/`*.hdr`) & optional binary ROI mask.
* **Output:** `AVCA_results.txt` – wavelengths + R endmember spectra.

The script internally estimates SNR and switches between projective & subspace projections as per \[Nascimento & Dias, 2005].

### 3️⃣ 3‑D CNN Classification & Abundance Mapping

```bash
$ python CNN_for_shaire.py \
      --rs       r"C:\path\prisma\scene.dat" \
      --hdr      r"C:\path\prisma\scene.hdr" \
      --mask     r"C:\path\mask.dat" \
      --sli_dir  r"C:\path\spectral_libraries" \
      --epochs   300 --batch_size 64
```

* **Input:**

  * PRISMA scene & HDR
  * Binary mask (1 = valid pixel)
  * Folder of augmented spectral libraries
* **Outputs:**

  * `classified.png` – mineral class map
  * `abundance.hdr/dat` – BSQ cube, one band per class
  * `training_log.csv` – accuracy / loss history

> **Early Stopping:** Training halts when both training & validation accuracy ≥ 0.993 and loss ≤ 0.11 (modifiable via `CustomEarlyStopping`).

---

## Expected Inputs & Directory Structure

```
├─ data/
│  ├─ prisma_scene.dat / .hdr
│  ├─ mask.dat / .hdr
│  └─ spectral_libraries/
│     ├─ Mineral_1.sli / .hdr
│     ├─ Mineral_2.sli / .hdr
│     └─ ...
```

Ensure consistent **interleave** (`bsq`) and **byte order** across all ENVI files.

---

## Outputs

| File                | Description                                                         |
| ------------------- | ------------------------------------------------------------------- |
| `Vegetation.txt`    | Augmented spectral library generated by Step 1.                     |
| `AVCA_results.txt`  | List of *R* endmember spectra extracted in Step 2.                  |
| `classified.png`    | Colour‑coded mineral map (includes “Unclassified”).                 |
| `abundance.hdr/dat` | Float32 BSQ cube; each band stores softmax probabilities per class. |

All outputs inherit geospatial metadata (map info & projection) from the input HDR to ensure GIS compatibility.

---

## Reproducibility Checklist

* [x] All random seeds (`numpy`, `tensorflow`) fixed where relevant.
* [x] Exact software versions specified in `requirements.txt`.
* [x] Training/validation split reported (85 / 15 %).
* [x] Model weights saved in‐memory; export hooks provided if persistence is required.

---

## Citation & License

This code is released under the permissive **MIT License** (see headers in each script).  If you use any part of this repository, **please cite our article**:

```bibtex
@article{Habashi2025DeepPRISMA,
  title     = {Revealing Critical Mineralogical Insights in Extreme Environments Using Deep Learning on PRISMA Hyperspectral Imagery: Dry Valleys, South Victoria Land, Antarctica},
  author    = {Habashi, Jabar and Beiranvand Pour, Amin and Muslim, Aidy M. and Moradi Afrapoli, Ali and Hong, Jong Kuk and Park, Yongcheol and Almasi, Alireza and Crispini, Laura and Hashim, Mazlan and Bagheri, Milad},
  journal   = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year      = {2025},
  doi       = {10.XXXX/XXXXX}
}
```

*Pending DOI will be updated upon publication.*

### Secondary citation requirement *(offset augmentation & AVCA)*

```
Silva, J. M. P. Nascimento & J. M. B. Dias, "Vertex Component Analysis: a Fast Algorithm to Unmix Hyperspectral Data," IEEE TGRS, 43(4):898–910, 2005.
```

---

## Contact & Support

* **Lead Developer:** Jabar Habashi – *jabar.habashi \[at] example.com*
* Issues and pull requests are very welcome! If you discover bugs or have suggestions for improvement, please open an [issue](https://github.com/<YourUser>/hyperspectral‑mineral‑mapping/issues) and fill out the template.

---

<div align="center">
✨ *Happy Mapping!* ✨
</div>
