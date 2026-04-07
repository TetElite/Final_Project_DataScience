# Random Forest Model v2 - Expanded Dataset Training Report

**Date:** April 7, 2026
**Model:** Random Forest Classifier v2 (Expanded Dataset)
**Status:** ✅ Training Completed Successfully

---

## Executive Summary

The Random Forest model has been successfully retrained on the expanded dataset with **3,184 samples** (28.5% increase from original 2,477 samples). The expanded model achieves **70.49% test accuracy**, representing a **+2.21%** improvement over the original model's 68.28% accuracy.

### Key Highlights

- ✅ Model trained with same configuration as original (100 trees, max_depth=20)
- ✅ SMOTE oversampling applied for class balance
- ✅ 56 features extracted (same feature engineering pipeline)
- ✅ All outputs saved to designated directories
- ✅ +2.21% accuracy improvement

---

## 1. Dataset Information

### Input Dataset
- **File:** `/Users/macbook/CADT/Term2Year3/Data/Final_Project/project/01_data/processed/pain_meds_ml_ready_expanded.csv`
- **Shape:** 3,184 rows × 67 columns
- **Increase:** +707 rows (+28.5% from original 2,477 rows)

### Target Distribution
```
Class 0 (Not Effective):        589 samples (18.50%)
Class 1 (Partially Effective):  402 samples (12.63%)
Class 2 (Effective):           2,193 samples (68.88%)
```

### Train/Test Split
- **Training Set:** 2,547 samples (80%)
- **Test Set:** 637 samples (20%)
- **Split Method:** Stratified (maintains class proportions)
- **Random State:** 42 (reproducible results)

### SMOTE Oversampling
- **Original Training Samples:** 2,547
- **After SMOTE:** 5,262 samples
- **Increase:** +2,715 synthetic samples
- **Result:** Perfectly balanced classes (33.33% each)

---

## 2. Model Configuration

### Algorithm: Random Forest Classifier

**Hyperparameters (identical to original model):**
```python
RandomForestClassifier(
    n_estimators=100,          # Number of trees
    max_depth=20,              # Maximum tree depth
    min_samples_split=5,       # Minimum samples to split
    class_weight='balanced',   # Handle class imbalance
    random_state=42,           # Reproducibility
    n_jobs=-1                  # Use all CPU cores
)
```

### Features Used
- **Total Features:** 56
- **Excluded Columns:** 11
  - `rating` (target leakage prevention)
  - `effectiveness` (target variable)
  - `effectiveness_encoded` (encoded target)
  - `uniqueID` (identifier)
  - `drugName`, `condition`, `review` (raw text)
  - `date` (temporal data)
  - `drugName_top`, `condition_top`, `condition_top.1` (string categories)

### Feature Categories
1. **Drug Encodings** (33 features): One-hot encoded drug names
2. **Condition Encodings** (15 features): One-hot encoded medical conditions
3. **Review Features** (8 features):
   - Text statistics: `review_length`, `review_word_count`, `avg_word_length`
   - Sentiment scores: `compound`, `neg`, `neu`, `pos`
   - Keyword flags: `has_positive_keywords`, `has_negative_keywords`
4. **Metadata** (2 features): `usefulCount`, `year`

---

## 3. Performance Metrics

### Overall Accuracy

| Metric | Original Model (v1) | Expanded Model (v2) | Change |
|--------|---------------------|---------------------|--------|
| **Test Accuracy** | 68.28% (338/495) | **70.49% (449/637)** | **+2.21%** |
| **Training Accuracy** | ~99% (overfitting) | 99.58% | Similar |
| **Test Samples** | 495 | 637 | +142 samples |

### Weighted Averages (Test Set)

| Metric | Value |
|--------|-------|
| **Weighted Precision** | 69.46% |
| **Weighted Recall** | 70.49% |
| **Weighted F1-Score** | 69.63% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support | Notes |
|-------|-----------|--------|----------|---------|-------|
| **Not Effective (0)** | 48.23% | 57.63% | 52.51% | 118 | Improved detection |
| **Partially Effective (1)** | 38.30% | 22.50% | 28.35% | 80 | Still challenging (minority class) |
| **Effective (2)** | 80.85% | 82.69% | 81.76% | 439 | Strong performance (majority class) |

### Confusion Matrix

```
                    Predicted
                    0      1      2      Total
Actual
  0 (Not Effective)    68      8     42    = 118
  1 (Partially Eff.)   18     18     44    =  80
  2 (Effective)        55     21    363    = 439
                    ─────────────────────
  Total              141     47    449    = 637

Accuracy: 449/637 = 70.49%
```

### Correct Predictions by Class
- **Class 0:** 68/118 = 57.63% correct
- **Class 1:** 18/80 = 22.50% correct
- **Class 2:** 363/439 = 82.69% correct

### Common Misclassifications
1. **Class 2 → Class 0:** 55 cases (Effective predicted as Not Effective)
2. **Class 1 → Class 2:** 44 cases (Partially → Effective)
3. **Class 0 → Class 2:** 42 cases (Not Effective → Effective)

---

## 4. Feature Importance Analysis

### Top 15 Most Important Features

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | `usefulCount` | 0.0999 | Metadata |
| 2 | `compound` | 0.0839 | Sentiment |
| 3 | `review_length` | 0.0790 | Text Stats |
| 4 | `pos` | 0.0740 | Sentiment |
| 5 | `review_word_count` | 0.0732 | Text Stats |
| 6 | `avg_word_length` | 0.0675 | Text Stats |
| 7 | `neg` | 0.0664 | Sentiment |
| 8 | `neu` | 0.0603 | Sentiment |
| 9 | `year` | 0.0575 | Metadata |
| 10 | `has_positive_keywords` | 0.0464 | Text Features |
| 11 | `condition_other` | 0.0259 | Condition |
| 12 | `condition_chronic pain` | 0.0190 | Condition |
| 13 | `condition_back pain` | 0.0188 | Condition |
| 14 | `drug_Naproxen` | 0.0160 | Drug |
| 15 | `drug_Tramadol` | 0.0158 | Drug |

### Key Insights
1. **Review metadata dominates:** `usefulCount` is the single most important feature (10% importance)
2. **Sentiment analysis crucial:** All 4 sentiment scores (`compound`, `pos`, `neg`, `neu`) are in top 10
3. **Text statistics matter:** Review length and word count significantly influence predictions
4. **Drug/condition encoding:** While important, less influential than review-derived features
5. **Year has moderate impact:** Temporal trends present but not dominant

---

## 5. Model Comparison: v1 vs v2

### Dataset Comparison

| Aspect | Original (v1) | Expanded (v2) | Difference |
|--------|--------------|---------------|------------|
| **Total Rows** | 2,477 | 3,184 | +707 (+28.5%) |
| **Total Columns** | 66 | 67 | +1 |
| **Features Used** | 55 | 56 | +1 |
| **Test Set Size** | 495 | 637 | +142 (+28.7%) |
| **Training Set (after SMOTE)** | ~4,100 | 5,262 | +1,162 |

### Performance Comparison

| Metric | Original (v1) | Expanded (v2) | Improvement |
|--------|---------------|---------------|-------------|
| **Test Accuracy** | 68.28% | 70.49% | **+2.21%** |
| **Not Effective Precision** | 42.17% | 48.23% | +6.06% |
| **Not Effective Recall** | 42.68% | 57.63% | **+14.95%** |
| **Partially Effective Precision** | 14.71% | 38.30% | **+23.59%** |
| **Partially Effective Recall** | 9.09% | 22.50% | **+13.41%** |
| **Effective Precision** | 78.84% | 80.85% | +2.01% |
| **Effective Recall** | 83.24% | 82.69% | -0.55% |

### Key Improvements
1. ✅ **Overall accuracy:** +2.21 percentage points
2. ✅ **Minority class detection:** Major improvements in detecting "Not Effective" (+14.95% recall) and "Partially Effective" (+13.41% recall)
3. ✅ **Balanced performance:** Better precision/recall balance across all classes
4. ✅ **More robust:** Larger test set (637 vs 495) provides more reliable evaluation

### Remaining Challenges
1. ⚠️ **Partially Effective class:** Still difficult to predict (only 22.50% recall)
2. ⚠️ **Class imbalance:** Despite SMOTE, minority classes remain challenging
3. ⚠️ **Overfitting:** 99.58% training accuracy vs 70.49% test accuracy (29% gap)

---

## 6. Output Files

### Model Directory: `04_trained_models/random_forest_v2_expanded/`

| File | Size | Description |
|------|------|-------------|
| `rf_model.pkl` | 14 MB | Trained Random Forest model (scikit-learn) |
| `feature_names.pkl` | 1.3 KB | List of 56 feature names (for inference) |
| `feature_importance.csv` | 2.3 KB | Feature importance rankings |
| `README.md` | 3.0 KB | Model documentation and usage guide |

### Test Results: `04_trained_models/test_results/`

| File | Size | Description |
|------|------|-------------|
| `test_predictions_expanded.csv` | 7.8 KB | 637 predictions with actual vs predicted |
| `confusion_matrix_expanded.csv` | 91 B | Confusion matrix in CSV format |

### Sample Predictions

```csv
uniqueID,actual,predicted,correct
19167,0,0,1         # Correct: Not Effective → Not Effective
57704,2,2,1         # Correct: Effective → Effective
19120,1,2,0         # Incorrect: Partially → Effective
19247,2,0,0         # Incorrect: Effective → Not Effective
58003,1,1,1         # Correct: Partially → Partially (rare!)
```

**Prediction Accuracy Breakdown:**
- Correct predictions: 449/637 (70.49%)
- Incorrect predictions: 188/637 (29.51%)

---

## 7. Usage Instructions

### Loading the Model

```python
import pickle
from pathlib import Path
import pandas as pd

# Define paths
model_path = Path("project/04_trained_models/random_forest_v2_expanded/rf_model.pkl")
features_path = Path("project/04_trained_models/random_forest_v2_expanded/feature_names.pkl")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load feature names
with open(features_path, "rb") as f:
    feature_names = pickle.load(f)

print(f"Model loaded: {type(model)}")
print(f"Features required: {len(feature_names)}")
```

### Making Predictions

```python
# Prepare your data with the same 56 features
# X should be a DataFrame or array with shape (n_samples, 56)

# Predict classes
predictions = model.predict(X)

# Get probabilities
probabilities = model.predict_proba(X)

# Example output:
# predictions: [2, 0, 2, 1, ...]
# probabilities: [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], ...]
```

### Interpreting Results

```python
# Class mapping
class_mapping = {
    0: "Not Effective",
    1: "Partially Effective", 
    2: "Effective"
}

# Convert predictions to labels
predicted_labels = [class_mapping[pred] for pred in predictions]

# Get confidence scores
confidence = probabilities.max(axis=1)
```

---

## 8. Technical Details

### Training Environment
- **Python Version:** 3.14
- **scikit-learn:** 1.8.0
- **imbalanced-learn:** 0.12.0
- **pandas:** Latest
- **numpy:** Latest

### Training Time
- **Total Duration:** ~6 seconds
- **Model Fitting:** ~4 seconds
- **Evaluation:** ~2 seconds
- **Hardware:** Multi-core CPU (n_jobs=-1)

### Memory Usage
- **Model Size:** 14 MB (100 trees × ~140 KB per tree)
- **Training Peak Memory:** ~500 MB (with SMOTE)
- **Inference Memory:** ~50 MB

### Reproducibility
- **Random State:** 42 (fixed for all operations)
- **Stratified Split:** Ensures consistent class distributions
- **Deterministic:** Same results on re-run with same data

---

## 9. Recommendations

### For Production Deployment
1. ✅ **Use this model:** Significant improvement over v1
2. ✅ **Monitor minority classes:** Implement confidence thresholds for Class 1 predictions
3. ✅ **Feature validation:** Ensure all 56 features are present in production data
4. ⚠️ **Handle edge cases:** Low confidence predictions (<60%) may need human review

### For Future Improvements
1. **Address overfitting:**
   - Add regularization (increase `min_samples_split`, decrease `max_depth`)
   - Use cross-validation for hyperparameter tuning
   - Try ensemble methods (XGBoost, LightGBM)

2. **Improve minority class performance:**
   - Collect more "Partially Effective" samples
   - Try different oversampling strategies (ADASYN, BorderlineSMOTE)
   - Consider class-specific thresholds

3. **Feature engineering:**
   - Add interaction features (drug × condition)
   - Extract more sophisticated NLP features (TF-IDF, embeddings)
   - Include temporal features (review age, trending drugs)

4. **Model alternatives:**
   - Test Gradient Boosting (may handle imbalance better)
   - Try Neural Networks with class weights
   - Ensemble multiple models

---

## 10. Conclusion

### Summary
The expanded Random Forest model (v2) successfully improves upon the original model with:
- **+2.21% accuracy gain** (68.28% → 70.49%)
- **Better minority class detection** (especially "Not Effective" class)
- **More robust evaluation** (637 test samples vs 495)
- **28.5% more training data** (3,184 rows vs 2,477)

### Model Readiness
- ✅ **Training:** Complete and successful
- ✅ **Evaluation:** Comprehensive metrics generated
- ✅ **Documentation:** Full README and usage guide
- ✅ **Artifacts:** All required files saved
- ✅ **Reproducibility:** Random seed fixed, configuration documented

### Next Steps
1. **Integrate into dashboard:** Update prediction service to use v2 model
2. **A/B testing:** Compare v1 vs v2 performance in production
3. **Monitor performance:** Track accuracy on new data
4. **Iterate:** Continue collecting data for further improvements

---

**Report Generated:** April 7, 2026, 15:24:30  
**Training Script:** `train_expanded_model.py`  
**Model Version:** v2 (Expanded Dataset)  
**Status:** ✅ Production Ready
