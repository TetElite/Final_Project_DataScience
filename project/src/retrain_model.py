import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import nltk

# Download NLTK data
print("Downloading NLTK data...")
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    print("✓ NLTK data downloaded")
except Exception as e:
    print(f"Warning: NLTK download issue: {e}")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "pain_meds_ml_ready.csv"
MODELS_DIR = BASE_DIR / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("RETRAINING RANDOM FOREST MODEL")
print("="*60)

# Load data
print(f"\nLoading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✓ Data loaded: {df.shape}")

# Check columns
print(f"\nColumns in dataset: {len(df.columns)}")
print(f"First 10 columns: {list(df.columns[:10])}")

# Define exclusions (columns NOT to use as features)
# NOTE: 'rating' is excluded to prevent data leakage since effectiveness_encoded
# is derived from rating bins. We want to predict effectiveness from review text,
# drug type, condition, and other legitimate features available BEFORE treatment.
exclude_cols = [
    'effectiveness', 'effectiveness_encoded', 'uniqueID',
    'drugName', 'condition', 'review', 'date',
    'drugName_top', 'condition_top',
    'rating'  # EXCLUDED: Causes target leakage
]

# Get feature columns (everything except exclusions)
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"\n✓ Using {len(feature_cols)} features")
print(f"Sample features: {feature_cols[:5]}")

# Prepare X and y
X = df[feature_cols]
y = df['effectiveness_encoded']

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nTarget distribution:")
print(y.value_counts().sort_index())
print("\nPercentages:")
print(y.value_counts(normalize=True).sort_index() * 100)

# Train/test split
print("\n" + "-"*60)
print("Splitting data (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
print(f"✓ Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"✓ Test size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Add SMOTE resampling to fix class imbalance
print("\n" + "-"*60)
print("Applying SMOTE to balance training data...")
print("\nOriginal training class distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print("\nPercentages:")
print(pd.Series(y_train).value_counts(normalize=True).sort_index() * 100)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nBalanced training class distribution (after SMOTE):")
print(pd.Series(y_train_balanced).value_counts().sort_index())
print("\nPercentages:")
print(pd.Series(y_train_balanced).value_counts(normalize=True).sort_index() * 100)
print(f"✓ Training samples increased: {len(X_train)} → {len(X_train_balanced)}")

# Train model with balanced class weights
print("\n" + "-"*60)
print("Training Random Forest with balanced class weights...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',  # Keep this for extra emphasis
    n_jobs=-1,
    max_depth=20,
    min_samples_split=5,
    verbose=1
)

print("Fitting model on BALANCED training data...")
model.fit(X_train_balanced, y_train_balanced)
print("✓ Model trained")

# Evaluate
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

y_train_pred = model.predict(X_train_balanced)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train_balanced, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nTraining Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

print("\n" + "-"*60)
print("CLASSIFICATION REPORT (Test Set)")
print("-"*60)
class_names = ['Not Effective (0)', 'Partially Effective (1)', 'Effective (2)']
print(classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0))

# Save model
print("\n" + "-"*60)
print("Saving model files...")
model_path = MODELS_DIR / "rf_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Model saved: {model_path}")

# Save feature names
feature_names_path = MODELS_DIR / "feature_names.pkl"
with open(feature_names_path, 'wb') as f:
    pickle.dump(feature_cols, f)
print(f"✓ Feature names saved: {feature_names_path}")

# Save test predictions
test_predictions = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_test_pred
})
test_pred_path = MODELS_DIR / "test_predictions.csv"
test_predictions.to_csv(test_pred_path, index=False)
print(f"✓ Test predictions saved: {test_pred_path}")

print("\n" + "="*60)
print("RETRAINING COMPLETE")
print("="*60)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
print(f"Model files saved to: {MODELS_DIR}")
print("\nReady for dashboard!")
