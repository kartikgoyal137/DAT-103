# Land Use and Land Cover (LULC) Classification

A pipeline for classifying Land Use and Land Cover using Landsat SR imagery and ESRI LULC data. The system utilizes deep learning to identify and categorize various geographical features, providing a robust tool for environmental monitoring and urban planning.

---

## Project Overview

The core of this project is a multi-stage data processing and machine learning pipeline:

- **Geospatial Analysis** — Processing Landsat Surface Reflectance (SR) imagery alongside ESRI's global land cover dataset.
- **Deep Learning** — Utilizing PyTorch to build and train models for precise feature classification across diverse terrain types.
- **Automated Pipeline** — A structured workflow covering everything from raw data ingestion to model evaluation and visualization.

---

## Setup Instructions

### 1. Environment Creation

Create and activate a virtual environment to ensure dependency isolation.

**Linux/macOS**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Python Version Verification

This project requires **Python 3.11.15** for full compatibility with the deep learning libraries used. Before proceeding, verify your version:

```bash
python --version
```

If the output is not exactly `Python 3.11.15`, please install the correct version from [python.org](https://www.python.org) before continuing.

### 3. Install Dependencies

Once the correct environment is active, install the required packages. This project relies on several key libraries including `torch`, `rasterio`, `rioxarray`, and `tqdm`:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
├── notebooks/
│   ├── Preprocessing.ipynb
│   ├── Data_Preparation.ipynb
│   └── Model.ipynb
├── utils/
│   └── helper.py
├── visualizations/
├── requirements.txt
└── README.md
```

- `notebooks/` — Contains the core logic for data processing and model training.
- `utils/helper.py` — A centralized script containing shared utility functions and class definitions used across the notebooks.
- `visualizations/` — Stores generated output patches and classification results for visual inspection.


---

## Execution Order

Execute the notebooks in the following sequence:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `Preprocessing.ipynb` | Handles initial data downloading, cleaning, and coordinate transformations using `rasterio` and `rioxarray`. |
| 2 | `Data_Preparation.ipynb` | Performs patch generation from large satellite tiles and applies data augmentation techniques to enhance training diversity. |
| 3 | `Model.ipynb` | Defines the neural network architecture, manages the training loop, and evaluates performance metrics like Precision and Recall. |
