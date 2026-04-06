# 🤖 TRAINED AI MODELS

## Location of Main Model

**Primary Model:** `random_forest_v1/rf_model.pkl` (11 MB)

### Model Details

- **Algorithm:** Random Forest Classifier
- **Accuracy:** 68.28% on test set
- **Features:** 55 input features
- **Classes:** 3 (Not Effective, Partially Effective, Effective)
- **Training Date:** April 2026
- **Framework:** scikit-learn 1.8.0

### Files in This Directory

```
04_trained_models/
├── random_forest_v1/
│   ├── rf_model.pkl              # ⭐ MAIN MODEL FILE (11 MB)
│   ├── feature_names.pkl         # List of 55 feature names
│   ├── feature_importance.csv    # Feature importance rankings
│   └── model_config.json         # Hyperparameters (optional)
└── test_results/
    ├── test_predictions.csv      # 495 test predictions
    ├── confusion_matrix.csv      # Performance matrix
    └── metrics_report.txt        # Accuracy, precision, recall, F1
```

## How to Load the Model

```python
import pickle
from pathlib import Path

# Load the model
model_path = Path("04_trained_models/random_forest_v1/rf_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load feature names
features_path = Path("04_trained_models/random_forest_v1/feature_names.pkl")
with open(features_path, "rb") as f:
    feature_names = pickle.load(f)

# Make prediction
# X should be a DataFrame with columns matching feature_names
prediction = model.predict(X)
probabilities = model.predict_proba(X)
```

## Model Performance

- **Overall Accuracy:** 68.28%
- **Weighted Precision:** 65.64%
- **Weighted Recall:** 68.28%
- **Weighted F1-Score:** 66.84%

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Effective | 42.17% | 42.68% | 42.42% | 82 |
| Partially Effective | 14.71% | 9.09% | 11.24% | 55 |
| Effective | 78.84% | 83.24% | 80.98% | 358 |

## Training Configuration

- n_estimators: 100
- max_depth: 20
- min_samples_split: 5
- class_weight: 'balanced'
- random_state: 42

## Retraining Instructions

To retrain the model with new data:

```bash
cd 03_source_code/model_training
python train_model.py --data ../../01_data/processed/pain_meds_ml_ready.csv
```

The new model will be saved to `04_trained_models/random_forest_v2/`
