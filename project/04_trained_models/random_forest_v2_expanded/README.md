# Random Forest Model v2 - Expanded Dataset

**Training Date:** 2026-04-07 15:24:19
**Dataset:** pain_meds_ml_ready_expanded.csv
**Dataset Size:** 3,184 rows × 67 columns

## Model Configuration

- **Algorithm:** Random Forest Classifier
- **n_estimators:** 100
- **max_depth:** 20
- **min_samples_split:** 5
- **class_weight:** 'balanced'
- **random_state:** 42
- **SMOTE applied:** Yes (k_neighbors=5)

## Dataset Split

- **Total samples:** 3,184
- **Training set:** 2,547 samples (80%)
- **Test set:** 637 samples (20%)
- **Training set after SMOTE:** 5,262 samples

## Features

- **Total features:** 56
- **Excluded columns:** 11 (rating, effectiveness, IDs, raw text)

### Top 10 Features by Importance

1. usefulCount (0.099974)
2. compound (0.083883)
3. review_length (0.078983)
4. pos (0.073970)
5. review_word_count (0.073171)
6. avg_word_length (0.067534)
7. neg (0.066360)
8. neu (0.060338)
9. year (0.057464)
10. has_positive_keywords (0.046413)

## Performance Metrics

### Overall Accuracy

- **Test Accuracy:** 70.49% (449/637 correct)
- **Training Accuracy:** 99.58%

### Weighted Averages (Test Set)

- **Weighted Precision:** 69.46%
- **Weighted Recall:** 70.49%
- **Weighted F1-Score:** 69.63%

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Effective (0) | 48.23% | 57.63% | 52.51% | 118 |
| Partially Effective (1) | 38.30% | 22.50% | 28.35% | 80 |
| Effective (2) | 80.85% | 82.69% | 81.76% | 439 |

### Confusion Matrix

```
           Predicted
           0      1      2
Actual
  0        68      8     42
  1        18     18     44
  2        55     21    363
```

## Comparison to Original Model (v1)

| Metric | Original (v1) | Expanded (v2) | Change |
|--------|---------------|---------------|--------|
| Dataset Size | 2,477 rows | 3,184 rows | +707 rows |
| Test Accuracy | 68.28% (338/495) | 70.49% (449/637) | +2.21% |
| Features | 55 | 56 | +1 |

## Files Saved

- `rf_model.pkl` - Trained Random Forest model
- `feature_names.pkl` - List of 56 feature names
- `feature_importance.csv` - Feature importance rankings
- `../test_results/test_predictions_expanded.csv` - Test predictions with actual vs predicted
- `../test_results/confusion_matrix_expanded.csv` - Confusion matrix

## Usage

```python
import pickle
from pathlib import Path

# Load model
model_path = Path("04_trained_models/random_forest_v2_expanded/rf_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load feature names
features_path = Path("04_trained_models/random_forest_v2_expanded/feature_names.pkl")
with open(features_path, "rb") as f:
    feature_names = pickle.load(f)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)
```

## Notes

- Rating column excluded to prevent target leakage
- SMOTE applied to balance training classes
- Model trained on balanced dataset but evaluated on original test distribution
- Test set maintains real-world class distribution
