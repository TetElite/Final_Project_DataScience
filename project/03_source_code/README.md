# 💻 SOURCE CODE

## Directory Structure

```
03_source_code/
├── data_processing/          # Data cleaning & feature engineering
│   ├── __init__.py
│   ├── data_loader.py        # Load raw datasets
│   ├── cleaning.py           # Clean text, handle missing values
│   └── feature_engineering.py # Create ML features
├── model_training/           # Model training & evaluation
│   ├── __init__.py
│   ├── train_model.py        # Train Random Forest
│   └── evaluate_model.py     # Calculate metrics
└── utils/                    # Helper functions
    ├── __init__.py
    └── visualization.py      # Plotting functions
```

## Usage

### Preprocess New Data
```bash
python data_processing/feature_engineering.py
```

### Train Model
```bash
python model_training/train_model.py
```

### Evaluate Performance
```bash
python model_training/evaluate_model.py
```

## Dependencies

See `../project_files/requirements.txt` for full list:
- pandas >= 1.5.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- nltk >= 3.8.1
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
