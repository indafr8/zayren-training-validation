# Zayren Training and Validation Project

This project consists of two main components: a training module and a validation module for machine learning model development and evaluation.

## Project Structure

```
.
├── data/               # Data directory for storing datasets
├── training/          # Training module
│   ├── train.py      # Main training script
│   ├── update.py     # Model update functionality
│   ├── processing.py # Data processing utilities
│   ├── past_data/    # Historical training data
│   └── output_data/  # Training outputs and models
├── validation/        # Validation module
│   ├── validation.py # Validation script
│   ├── api_local/    # Local API implementation
│   ├── output_data/  # Validation results
│   ├── base_data_0.csv
│   └── new_data_0.csv
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- TensorFlow 2.10.0
- NumPy 1.23.5
- Pandas 1.5.3
- Scikit-learn 1.2.2

## Components

### Training Module
The training module (`training/`) contains scripts for model training and data processing:
- `train.py`: Main script for model training
- `update.py`: Handles model updates and retraining
- `processing.py`: Data preprocessing and feature engineering utilities

### Validation Module
The validation module (`validation/`) handles model validation and testing:
- `validation.py`: Main validation script
- `api_local/`: Local API implementation for model serving
- Contains test datasets (`base_data_0.csv` and `new_data_0.csv`)

## Usage

1. Training:
```bash
python training/train.py
```

2. Validation:
```bash
python validation/validation.py
```

## Data
- Training data is stored in `training/past_data/`
- Validation data is stored in `validation/`
- Processed outputs are saved in respective `output_data/` directories