# Frustrated Lewis Pairs for H₂ Splitting  

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
![Conda](https://img.shields.io/badge/conda-environment-green.svg)  
![Status](https://img.shields.io/badge/status-active-success.svg)  

This repository provides Python scripts to extract, process, and analyze data from Gaussian output files in order to assess correlations between the catalytic performance of Frustrated Lewis Pairs (FLPs) in H₂ splitting and key molecular descriptors.

---

## Table of Contents
- [About](#about)  
- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Contact](#contact)

---

## About

This project investigates the catalytic reactivity of Frustrated Lewis Pairs (FLPs) for hydrogen (H₂) splitting. FLPs are metal-free molecular systems capable of activating small molecules such as H₂ and CO₂, making them promising candidates for sustainable catalysis.

The core of this work focuses on a curated dataset of **112 intramolecular boron/nitrogen (B/N) FLPs**, systematically designed to explore the relationship between molecular structure and catalytic performance.  

- **Thermodynamic (ΔG)** and **kinetic (ΔG‡)** parameters were computed using Density Functional Theory (DFT).  
- **Partial Least Squares (PLS)** regression methods were applied to derive chemically meaningful structure–activity relationships.  

### Key outcomes
- Identification of geometric and electronic descriptors that govern H₂ activation.  
- Predictive models for both reaction thermodynamics and activation barriers.  
- Insights into how acid/base strength and preorganization influence catalytic efficiency.  
- A framework for rational FLP design based on transferable molecular descriptors.  

This repository provides computational tools to guide the development of next-generation FLP catalysts for efficient H₂ splitting.

---

## Features

- **Data extraction** from Gaussian output files (`data_extractor.py`).  
- **Automatic detection of FLPs** (Lewis acid/base centers) (`flp_detector.py`).  
- **Descriptor generation**: geometric and electronic features (`desc_gen.py`).  
- **PLS model construction** for structure–activity relationships (`pls_analysis.py`).  
- **Descriptor expansion** (e.g., squared terms for flexible models) (`data_expander.py`).  
- **Informative variable selection** to optimize models without losing predictive ability (`informative_var_filter_PLS_iter.py`).  
- **External validation** of optimized models with independent datasets (`model_validation.py`).  
- **Transition state analysis**:  
  - H–H bond distance calculation (`h2_ts_dist_calc.py`)  
  - Distortion energy evaluation and Gaussian input generation (`file_generation_dist_ener.py`)  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/FrancescPenas/FLPs_for_H2_splitting.git
   cd FLPs_for_H2_splitting
   ```

2. Create the environment with **Anaconda**:
   ```bash
   conda env create -f environment.yml -n my_custom_name
   conda activate my_custom_name
   ```

This environment includes all required Python modules with compatible versions.

---

## Usage

Below is a summary of the main scripts. Each follows a **Purpose → Inputs → Outputs** structure for clarity.

---

### `data_extractor.py`
**Purpose:** Extract molecular information from Gaussian output files.  
**Inputs:**  
- `input_dir`: Directory containing Gaussian output files.  
- `esp_charges_dir`: Directory with ChelpG charges (added later).  

**Process:**  
- Reads Gaussian outputs and extracts:  
  - Coordinates (`gaussian_coordinates_extractor`)  
  - NBO orbitals and charges (`gaussian_nbo_extractor`, `gaussian_nat_charges_extractor`)  
  - Atomic connectivity (`gaussian_connectivity_extractor`)  
  - ESP charges (`gaussian_esp_charges_extractor`)  

**Outputs:**  
- `input_data.pkl`: Extracted data.  
- `data_extractor_error_list.csv`: Log of extraction errors.  

---

### `flp_detector.py`
**Purpose:** Identify Lewis acid/base centers in molecules.  
**Process:**  
1. Detect Lewis acids (boron lone pair acceptors) via `gaussian_la_detector`.  
2. Detect Lewis bases (nitrogen lone pairs) via `gaussian_lb_detector`, excluding directly bonded atoms and selecting the closest candidate.  

**Outputs:**  
- `data_flp.pkl`: Dataset with FLP centers identified, and their distance.  

---

### `desc_gen.py`
**Purpose:** Generate chemically relevant descriptors for each molecule.  
**Process:**  
- Calculate orbital energies, distances, midpoints, molecular weight, and angles.  
- Compute charges, electric fields, and electrostatic potentials.  
- Import external hydride/proton attachment energies (`feha_energies.csv`, `fepa_energies.csv`).  
- Incorporate computed Gibbs free energies (`h_split_ener.csv`, `h_split_ener_barr.csv`).  

**Outputs:**  
- `final_data.pkl` (pickle format).  
- `final_data.csv` (CSV format).  

---

### `pls_analysis.py`
**Purpose:** Build and validate PLS models.  
**Process:**  
1. Normalize descriptors with `MinMaxScaler`.  
2. Reduce dimensionality with `PLSRegression`.  
3. Fit models and predict responses.  
4. Perform k-fold cross-validation (optimized to 60 folds).  
5. Evaluate with r², q², and RMSE.  

**Outputs:**  
- Model coefficients (normalized & rescaled).  
- Prediction results and model statistics.  

---

### `data_expander.py`
**Purpose:** Enhance descriptor set by adding polynomial terms.  
**Process:**  
- Square selected descriptors and add them to the dataset.  
**Outputs:**  
- `final_data_expanded.pkl`  
- `final_data_expanded.csv`  

---

### `informative_var_filter_PLS_iter.py`
**Purpose:** Select informative descriptors and filter noise.  
**Method:** Based on Andries *et al.*, *Analytica Chimica Acta* 2017, 982, 37–47.  
**Process:**  
- Iteratively filter descriptors by comparing impact coefficients to randomized variables.  
- Limit to max 15 descriptors for efficiency.  
- Build optimized models and evaluate.  

**Outputs:**  
- Iterative analysis results (`iter_{iteration}_barr_coefs.pkl`).  
- Optimized models (`opt_barr_models.pkl`, `opt_barr_models.csv`).  

---

### `model_validation.py`
**Purpose:** Validate models using independent FLP datasets.  
**Inputs:**  
- Validation sets:  
  - `intra_B_N_data.pkl`, `intra_Al_N_data.pkl`, `intra_B_P_data.pkl`, `intra_Al_P_data.pkl`, `inter_B_N_data.pkl`  

**Process:**  
- Load model and validation set.  
- Randomize response variable (control test).  
- Predict external validation set and calculate metrics.  

**Outputs:**  
- `combined_results`: DataFrame with original vs. predicted values and model statistics.  

---

### `h2_ts_dist_calc.py`
**Purpose:** Calculate H–H bond distances in transition states.  
**Process:**  
- Extract vibrational frequencies via `gaussian_freq_extractor.py`.  
- Identify imaginary frequency and corresponding H atoms.  
- Calculate H–H distances from coordinates.  

**Outputs:**  
- `input_ts_data.pkl` with computed distances.  

---

### `file_generation_dist_ener.py`
**Purpose:** Generate Gaussian input files for distortion energy calculations.  
**Process:**  
- Identify H atoms involved in H–H cleavage.  
- Remove them and prepare Gaussian input files for distortion energy evaluation.  
- Generate corresponding PDB files for visualization.  

---

## Contact
For questions or collaborations, please contact:  
**Francesc Penas Hidalgo**  
✉️ [francesc.penas-hidalgo@college-de-france.fr]
✉️ [penashidalgo@gmail.com]