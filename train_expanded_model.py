import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("TRAINING RANDOM FOREST MODEL - EXPANDED DATASET (v2)")
print("="*80)
print(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "project" / "01_data" / "processed" / "pain_meds_ml_ready_expanded.csv"
OUTPUT_DIR = BASE_DIR / "project" / "04_trained_models" / "random_forest_v2_expanded"
TEST_RESULTS_DIR = BASE_DIR / "project" / "04_trained_models" / "test_results"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print(f"\n{'='*80}")
print("STEP 1: LOADING DATA")
print(f"{'='*80}")
print(f"Data path: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✓ Data loaded successfully")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Check for string columns that should be excluded
string_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nString columns detected: {string_cols}")

# Define exclusions (same as original model)
exclude_cols = [
    'effectiveness',           # Target variable name
    'effectiveness_encoded',   # Actual target
    'uniqueID',               # ID column
    'drugName',               # Categorical text (encoded versions used)
    'condition',              # Categorical text (encoded versions used)
    'review',                 # Raw text (features extracted)
    'date',                   # Date column
    'drugName_top',           # Text version of drug
    'condition_top',          # Text version of condition
    'condition_top.1',        # Duplicate condition column
    'rating'                  # EXCLUDED: Causes target leakage
]

# Add any additional string columns to exclusions
for col in string_cols:
    if col not in exclude_cols:
        exclude_cols.append(col)
        print(f"  Additional string column excluded: {col}")

# Get feature columns
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"\n✓ Feature selection complete")
print(f"  Total features: {len(feature_cols)}")
print(f"  Excluded columns: {len(exclude_cols)}")

# Display sample features
print(f"\n  Sample features (first 10):")
for i, feat in enumerate(feature_cols[:10], 1):
    print(f"    {i}. {feat}")

# Prepare X and y
print(f"\n{'='*80}")
print("STEP 2: PREPARING FEATURES AND TARGET")
print(f"{'='*80}")

X = df[feature_cols].copy()
y = df['effectiveness_encoded'].copy()

print(f"✓ Feature matrix: {X.shape}")
print(f"✓ Target vector: {y.shape}")

# Check for missing values
missing_counts = X.isnull().sum()
if missing_counts.sum() > 0:
    print(f"\n⚠ Warning: Found {missing_counts.sum()} missing values")
    print(f"  Filling missing values with 0...")
    X = X.fillna(0)
else:
    print(f"✓ No missing values found")

# Target distribution
print(f"\n{'='*80}")
print("TARGET DISTRIBUTION")
print(f"{'='*80}")
target_dist = y.value_counts().sort_index()
target_pct = y.value_counts(normalize=True).sort_index() * 100

print("\nClass Distribution:")
print(f"  0 (Not Effective):        {target_dist[0]:>5} ({target_pct[0]:>5.2f}%)")
print(f"  1 (Partially Effective):  {target_dist[1]:>5} ({target_pct[1]:>5.2f}%)")
print(f"  2 (Effective):            {target_dist[2]:>5} ({target_pct[2]:>5.2f}%)")
print(f"  Total:                    {len(y):>5}")

# Train/test split (80/20, same as original)
print(f"\n{'='*80}")
print("STEP 3: TRAIN/TEST SPLIT")
print(f"{'='*80}")
print(f"Split ratio: 80/20")
print(f"Stratified: Yes")
print(f"Random state: 42")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\n✓ Split complete")
print(f"  Training set:   {len(X_train):>5} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test set:       {len(X_test):>5} samples ({len(X_test)/len(X)*100:.1f}%)")

# Display training set class distribution
train_dist = pd.Series(y_train).value_counts().sort_index()
train_pct = pd.Series(y_train).value_counts(normalize=True).sort_index() * 100

print(f"\nTraining set class distribution:")
print(f"  0 (Not Effective):        {train_dist[0]:>5} ({train_pct[0]:>5.2f}%)")
print(f"  1 (Partially Effective):  {train_dist[1]:>5} ({train_pct[1]:>5.2f}%)")
print(f"  2 (Effective):            {train_dist[2]:>5} ({train_pct[2]:>5.2f}%)")

# Apply SMOTE
print(f"\n{'='*80}")
print("STEP 4: APPLYING SMOTE (OVERSAMPLING)")
print(f"{'='*80}")
print(f"Purpose: Balance class distribution in training set")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

balanced_dist = pd.Series(y_train_balanced).value_counts().sort_index()
balanced_pct = pd.Series(y_train_balanced).value_counts(normalize=True).sort_index() * 100

print(f"\n✓ SMOTE applied successfully")
print(f"  Original training samples: {len(X_train):>5}")
print(f"  Balanced training samples: {len(X_train_balanced):>5}")
print(f"  Increase: +{len(X_train_balanced) - len(X_train):>5} samples")

print(f"\nBalanced class distribution:")
print(f"  0 (Not Effective):        {balanced_dist[0]:>5} ({balanced_pct[0]:>5.2f}%)")
print(f"  1 (Partially Effective):  {balanced_dist[1]:>5} ({balanced_pct[1]:>5.2f}%)")
print(f"  2 (Effective):            {balanced_dist[2]:>5} ({balanced_pct[2]:>5.2f}%)")

# Train model (same configuration as original)
print(f"\n{'='*80}")
print("STEP 5: TRAINING RANDOM FOREST MODEL")
print(f"{'='*80}")
print("\nModel Configuration:")
print(f"  Algorithm:         Random Forest Classifier")
print(f"  n_estimators:      100")
print(f"  max_depth:         20")
print(f"  min_samples_split: 5")
print(f"  class_weight:      'balanced'")
print(f"  random_state:      42")
print(f"  n_jobs:            -1 (use all CPU cores)")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print(f"\nTraining in progress...")
model.fit(X_train_balanced, y_train_balanced)
print(f"✓ Model training complete")

# Make predictions
print(f"\n{'='*80}")
print("STEP 6: MODEL EVALUATION")
print(f"{'='*80}")

y_train_pred = model.predict(X_train_balanced)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train_balanced, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

correct_predictions = (y_test == y_test_pred).sum()
total_predictions = len(y_test)

print(f"\nAccuracy Scores:")
print(f"  Training Accuracy:  {train_acc*100:.2f}% ({int(train_acc*len(y_train_balanced))}/{len(y_train_balanced)} correct)")
print(f"  Test Accuracy:      {test_acc*100:.2f}% ({correct_predictions}/{total_predictions} correct)")

# Classification report
print(f"\n{'='*80}")
print("CLASSIFICATION REPORT (Test Set)")
print(f"{'='*80}")

class_names = ['Not Effective (0)', 'Partially Effective (1)', 'Effective (2)']
print("\n" + classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))

# Confusion matrix
print(f"{'='*80}")
print("CONFUSION MATRIX (Test Set)")
print(f"{'='*80}")

cm = confusion_matrix(y_test, y_test_pred)
print("\n           Predicted")
print("           0      1      2")
print("Actual")
for i, row in enumerate(cm):
    print(f"  {i}     {row[0]:>5}  {row[1]:>5}  {row[2]:>5}")

# Feature importance
print(f"\n{'='*80}")
print("STEP 7: FEATURE IMPORTANCE")
print(f"{'='*80}")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
for idx, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
    print(f"  {idx:>2}. {row['feature']:<40} {row['importance']:.6f}")

# Save outputs
print(f"\n{'='*80}")
print("STEP 8: SAVING MODEL AND RESULTS")
print(f"{'='*80}")

# Save model
model_path = OUTPUT_DIR / "rf_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Model saved: {model_path}")

# Save feature names
feature_names_path = OUTPUT_DIR / "feature_names.pkl"
with open(feature_names_path, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"✓ Feature names saved: {feature_names_path}")

# Save feature importance
importance_path = OUTPUT_DIR / "feature_importance.csv"
feature_importance.to_csv(importance_path, index=False)
print(f"✓ Feature importance saved: {importance_path}")

# Save test predictions with original IDs
test_predictions = pd.DataFrame({
    'uniqueID': df.iloc[y_test.index]['uniqueID'].values,
    'actual': y_test.values,
    'predicted': y_test_pred,
    'correct': (y_test.values == y_test_pred).astype(int)
})
test_pred_path = TEST_RESULTS_DIR / "test_predictions_expanded.csv"
test_predictions.to_csv(test_pred_path, index=False)
print(f"✓ Test predictions saved: {test_pred_path}")

# Save confusion matrix
cm_df = pd.DataFrame(cm, 
                     columns=['Predicted_0', 'Predicted_1', 'Predicted_2'],
                     index=['Actual_0', 'Actual_1', 'Actual_2'])
cm_path = TEST_RESULTS_DIR / "confusion_matrix_expanded.csv"
cm_df.to_csv(cm_path)
print(f"✓ Confusion matrix saved: {cm_path}")

# Generate comprehensive README
print(f"✓ Generating metrics report...")

# Calculate per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_test_pred, average=None, zero_division=0
)

# Calculate weighted averages
precision_weighted = precision_recall_fscore_support(
    y_test, y_test_pred, average='weighted', zero_division=0
)[0]
recall_weighted = precision_recall_fscore_support(
    y_test, y_test_pred, average='weighted', zero_division=0
)[1]
f1_weighted = precision_recall_fscore_support(
    y_test, y_test_pred, average='weighted', zero_division=0
)[2]

# Original model metrics for comparison
original_acc = 68.28
original_correct = 338
original_total = 495

# Calculate improvement
acc_improvement = test_acc * 100 - original_acc
acc_improvement_sign = "+" if acc_improvement > 0 else ""

readme_content = f"""# Random Forest Model v2 - Expanded Dataset

**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** pain_meds_ml_ready_expanded.csv
**Dataset Size:** {df.shape[0]:,} rows × {df.shape[1]} columns

## Model Configuration

- **Algorithm:** Random Forest Classifier
- **n_estimators:** 100
- **max_depth:** 20
- **min_samples_split:** 5
- **class_weight:** 'balanced'
- **random_state:** 42
- **SMOTE applied:** Yes (k_neighbors=5)

## Dataset Split

- **Total samples:** {len(df):,}
- **Training set:** {len(X_train):,} samples (80%)
- **Test set:** {len(X_test):,} samples (20%)
- **Training set after SMOTE:** {len(X_train_balanced):,} samples

## Features

- **Total features:** {len(feature_cols)}
- **Excluded columns:** {len(exclude_cols)} (rating, effectiveness, IDs, raw text)

### Top 10 Features by Importance

{chr(10).join([f"{i}. {row['feature']} ({row['importance']:.6f})" for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1)])}

## Performance Metrics

### Overall Accuracy

- **Test Accuracy:** {test_acc*100:.2f}% ({correct_predictions}/{total_predictions} correct)
- **Training Accuracy:** {train_acc*100:.2f}%

### Weighted Averages (Test Set)

- **Weighted Precision:** {precision_weighted*100:.2f}%
- **Weighted Recall:** {recall_weighted*100:.2f}%
- **Weighted F1-Score:** {f1_weighted*100:.2f}%

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Effective (0) | {precision[0]*100:.2f}% | {recall[0]*100:.2f}% | {f1[0]*100:.2f}% | {support[0]} |
| Partially Effective (1) | {precision[1]*100:.2f}% | {recall[1]*100:.2f}% | {f1[1]*100:.2f}% | {support[1]} |
| Effective (2) | {precision[2]*100:.2f}% | {recall[2]*100:.2f}% | {f1[2]*100:.2f}% | {support[2]} |

### Confusion Matrix

```
           Predicted
           0      1      2
Actual
  0     {cm[0][0]:>5}  {cm[0][1]:>5}  {cm[0][2]:>5}
  1     {cm[1][0]:>5}  {cm[1][1]:>5}  {cm[1][2]:>5}
  2     {cm[2][0]:>5}  {cm[2][1]:>5}  {cm[2][2]:>5}
```

## Comparison to Original Model (v1)

| Metric | Original (v1) | Expanded (v2) | Change |
|--------|---------------|---------------|--------|
| Dataset Size | 2,477 rows | {df.shape[0]:,} rows | +{df.shape[0] - 2477:,} rows |
| Test Accuracy | {original_acc}% ({original_correct}/{original_total}) | {test_acc*100:.2f}% ({correct_predictions}/{total_predictions}) | {acc_improvement_sign}{acc_improvement:.2f}% |
| Features | 55 | {len(feature_cols)} | {len(feature_cols) - 55:+d} |

## Files Saved

- `rf_model.pkl` - Trained Random Forest model
- `feature_names.pkl` - List of {len(feature_cols)} feature names
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
"""

readme_path = OUTPUT_DIR / "README.md"
with open(readme_path, 'w') as f:
    f.write(readme_content)
print(f"✓ Metrics report saved: {readme_path}")

# Print final summary
print(f"\n{'='*80}")
print("TRAINING COMPLETE - SUMMARY")
print(f"{'='*80}")
print(f"\n📊 DATASET:")
print(f"   Expanded Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Original Dataset: 2,477 rows")
print(f"   Increase: +{df.shape[0] - 2477:,} rows ({(df.shape[0] - 2477) / 2477 * 100:.1f}%)")

print(f"\n🎯 TEST ACCURACY:")
print(f"   Original Model (v1):  {original_acc:.2f}% ({original_correct}/{original_total} correct)")
print(f"   Expanded Model (v2):  {test_acc*100:.2f}% ({correct_predictions}/{total_predictions} correct)")
print(f"   Improvement:          {acc_improvement_sign}{acc_improvement:.2f}%")

print(f"\n📁 OUTPUT FILES:")
print(f"   Model Directory:      {OUTPUT_DIR}")
print(f"   - rf_model.pkl")
print(f"   - feature_names.pkl")
print(f"   - feature_importance.csv")
print(f"   - README.md")
print(f"\n   Test Results:         {TEST_RESULTS_DIR}")
print(f"   - test_predictions_expanded.csv")
print(f"   - confusion_matrix_expanded.csv")

print(f"\n{'='*80}")
print("✅ ALL TASKS COMPLETED SUCCESSFULLY")
print(f"{'='*80}\n")
