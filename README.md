# plugin-mtsad
Plug-in module for multivariate time-series anomaly detection, designed to improve detection performance and enable interpretable analysis of inter-variable dependencies.

# A Plug-in Module for Detecting and Explaining Anomalies in Inter-Variable Dependencies in Multivariate Time Series

This repository contains the implementation of the plug-in module proposed in the paper  
**“A Plug-in Module for Detecting and Explaining Anomalies in Inter-Variable Dependencies in Multivariate Time Series.”**

The proposed module enhances existing multivariate time-series anomaly detection models by explicitly detecting and explaining anomalies in inter-variable dependency structures.

---

## Overview

Most multivariate time-series anomaly detection models implicitly capture inter-variable dependencies
(e.g., through attention or graph-based message passing), but they do not explicitly assess whether these learned dependencies deviate from normal patterns.

Our approach introduces a lightweight **plug-in module** that:

- Extracts inter-variable dependency (attention) matrices from a base model which could explicitly learns inter-variable dependency 
- Learns normal dependency patterns using an autoencoder trained on normal data  
- Quantifies dependency deviations via reconstruction error  
- Combines dependency-based scores with the base model’s anomaly score  
- Provides interpretable explanations of anomalous dependency changes  

The plug-in can be attached to existing models without modifying their original architectures.

---

## Dependencies

- numpy  
- pandas  
- torch  
- scikit-learn  
- scipy  

---

## File Descriptions

- `main.py`  
  Parses experiment arguments(e.g., --dataset, --topk, --lambda_attn, --train/--test), builds the dataset and base model, and runs training and/or testing pipelines.

- `dataset.py`  
  Loads raw multivariate time-series data and constructs sliding-window datasets(`TimeDataset`) for training, validation, and testing.
  
- `model.py`
  Implements the GDN base model integrated with the dependency-aware plug-in for anomaly detection and explanation.  

- `plugin_module.py`
  Defines the dependency-aware plug-in module(`plugin_module`) and extracts dense inter-variable dependency matrices from base model(`extract_dependency_matrix``).

- `train.py`
  Implements the training loop, validation-based early stopping(`train`), and test-time inference logic(`test`).

- `anomaly_detection.py`
  Computes anomaly scores and evaluates detection performance using point-wise and range-wise(point-adjusted) metrics (`best_threshold_rangewise_pa`).
  
- `explanation.ipynb`
  Visualizes and analyzes reconstructed dependency matrices to explain inter-variable anomaly patterns.
---

## Usage

1. Place dataset CSV files under the `datasets/` directory.
2. Select a base model and configure plug-in parameters.
3. Run the experiment:
   ```bash
   python main.py --dataset swat --lambda_attn 0.1 --topk 15 --random_seed 41 --train --test

