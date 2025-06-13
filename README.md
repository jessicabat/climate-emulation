# Climate Model Emulation with Hybrid U-Net FNO

### Jessica Batbayar

## Project Overview

This project focuses on developing a deep learning model to accurately emulate a physics-based climate model, specifically to project future climate patterns under varying emissions scenarios. The goal is to significantly reduce the computational cost associated with traditional climate simulations while maintaining high prediction accuracy for key climate variables like temperature and precipitation. My solution leverages a novel **Hybrid U-Net and Fourier Neural Operator (FNO) architecture**, iteratively refined to achieve strong performance.

## Table of Contents

1.  [Project Goal](#1-project-goal)
2.  [Dataset](#2-dataset)
3.  [Model Development: Final Architecture](#3-model-development-final-architecture)
4.  [Key Engineering & Optimization](#4-key-engineering--optimization)
5.  [Final Model Performance Summary](#5-final-model-performance-summary)
6.  [Limitations & Future Work](#6-limitations--future-work)

---

## 1. Project Goal

The primary objective was to build a machine learning emulator capable of predicting monthly climate variables (surface air temperature - `tas`, and precipitation - `pr`) given various input forcings, such as greenhouse gas concentrations and aerosols. The challenge was to accurately capture both spatial patterns and temporal variability, simulating out-of-distribution future climate scenarios effectively.

---

## 2. Dataset

The project utilizes data from **CMIP6 climate model simulations** under different Shared Socioeconomic Pathway (SSP) scenarios, provided in `Zarr` format.

* **Inputs:** `CO2`, `SO2`, `CH4`, `BC` (Black Carbon), `rsdt` (incoming shortwave radiation).
* **Outputs:** `tas` (Surface Air Temperature), `pr` (Precipitation).
* **Target Member ID:** `0` (a specific ensemble member).
* **Time & Spatial Resolution:** Monthly data on a global gridded resolution.

### Data Splitting Strategy:

* **Training Data:** Concatenated data from `ssp126`, `ssp585`, and the majority of `ssp370`.
* **Validation Data:** The **last 120 months of `ssp370`**. This served as an in-domain future projection to monitor generalization during training.
* **Test Data:** The **last 120 months of `ssp245`**. This represented a crucial **out-of-distribution scenario**, rigorously testing the model's ability to generalize to unseen future emissions pathways.

### Data Normalization:

All input and output variables underwent **Z-score normalization** (mean 0, standard deviation 1). Critically, normalization statistics were computed *exclusively from the training dataset* to prevent data leakage. Predictions were inverse-transformed for evaluation.

---

## 3. Model Development: Final Architecture

The final model is a **3-step Hybrid U-Net + FNO** architecture, representing the culmination of an iterative development process that explored pure U-Net and FNO models, followed by hybrid designs.

* **Motivation:** This architecture emerged from the need to combine the U-Net's strengths in capturing **multi-scale spatial features** with the FNO's efficiency in modeling **global spectral relationships**, and then further enhance depth for improved accuracy.
* **Architecture:** The model features a **3-step U-Net encoder** for deep spatial feature extraction, an **FNO block** (`modes=32`, `width=128`) integrated at the deepest `H/8 x W/8` bottleneck, and a corresponding **3-step U-Net decoder** for reconstruction.

---

## 4. Key Engineering & Optimization

Critical engineering decisions enabled the training and performance of the final model:

* **Architectural Deepening:** The deliberate shift from a 2-step to a **3-step U-Net encoder/decoder** significantly boosted model capacity and prediction accuracy.
* **Robust Memory Management:** Essential for accommodating the deeper architecture, this involved leveraging **Mixed-Precision Training (FP16)** and carefully tuning the `batch_size` to **16**.
* **Optimized Training Dynamics:** Implementation of the **OneCycleLR learning rate scheduler** (`max_lr=7e-4`, `pct_start=0.1`, `div_factor=15`) ensured fast and stable convergence.
* **Gradient Clipping:** Applied (`gradient_clip_val=1.0`) to prevent exploding gradients and maintain training stability in the deep network.
* **Seamless Architectural Integration:** The FNO's `width` was precisely matched to the U-Net's `bottle_channels` (both 128), streamlining feature flow and enhancing efficiency.

---

## 5. Final Model Performance Summary

The final 3-step Hybrid U-Net FNO model achieved the following best validation RMSEs, representing a significant improvement over earlier iterations:

| Metric                          | `tas` (Temperature) | `pr` (Precipitation) |
| :------------------------------ | :------------------ | :------------------- |
| **Validation RMSE** | **1.3100** | **1.9845** |
| Validation Time-Mean RMSE       | 0.4473              | 0.2265               |
| Validation Time-Stddev MAE      | 0.2500              | 0.7560               |

---

## 6. Limitations & Future Work

### Limitations:

* **`pr` Prediction Gap:** Despite strong overall performance, `pr` prediction did not see as significant improvements as `tas`, indicating remaining challenges with precipitation's complex variability.
* **Limited Data Exploration:** The project primarily relied on the provided data processing pipeline, with less focus on extensive custom feature engineering.
* **Computational Costs:** Training and inference times can be lengthy, limiting the scope for more exhaustive experimentation.

### Future Work:

* **Earlier Start & Broader Exploration:** Allocate more time for wider exploration of model architectures, hyperparameter spaces, and iterative refinement.
* **Creative Data Engineering:** Investigate advanced feature engineering (e.g., explicit temporal context, multi-modal inputs, alternative spatial augmentations) to better capture climate nuances.
* **Targeted `pr` Enhancement:** Implement specific strategies for precipitation, such as alternative loss functions, architectural modifications tailored to `pr`'s characteristics, or incorporating additional relevant input variables.
* **Ensemble Methods:** Explore combining predictions from multiple trained models to potentially improve robustness and accuracy.
