# ✈️ Project: Variational Graph Autoencoder (VGAE) for Airport Data (Link Prediction)

This project implements a **Variational Graph Autoencoder (VGAE)** using PyTorch Geometric to predict missing links (flights/connections) within a graph representing global airports. The model learns a low-dimensional embedding for each airport node based on its structural connections and features (geographic coordinates, population, and country).

## 🚀 Key Components and Files

- `main.ipynb` The main execution notebook. It coordinates the workflow: data loading, model setup, multi-run training, threshold search, and displaying final results.
- `preprocessing.py` Handles all data preparation: graph loading (.graphml), cleaning (removing low-degree nodes), feature engineering (normalization, One-Hot Encoding), and train/validation/test edge splitting.
- `model.py` Defines the 2-layer GCN Encoder for the VGAE architecture.
- `train_eval.py` Contains the core logic for training (including the $\text{BCE} + \text{KL}$ loss calculation) and evaluation (AUC, AP). Implements early stopping and multi-run execution.
- `config.py` Stores global settings (device, seed) and utility functions (loss calculation, metric aggregation, graph reconstruction logic).
- `visualization.py` Manages all plotting tasks: training curves, score distributions, Precision-Recall curves, and the final geographical map comparison using cartopy.

## 🛠️ Installation

The project relies on PyTorch, PyTorch Geometric, NetworkX, and geographic plotting libraries (`matplotlib` and `cartopy`). The recommended way to set up the environment is using Conda with the provided `environment.yml` file.

**Step-by-Step Installation**

1. **Create and Activate Conda Environment:** Open your terminal or Conda prompt and run the following commands.

```bash
# 1. Create the environment using the YAML file
conda env create -f environment.yml

# 2. Activate the newly created environment
conda activate gnn_project
```

2. **Run `main.ipynb`**
