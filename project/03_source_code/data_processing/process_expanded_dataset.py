"""
Process Expanded Pain Medication Dataset Through Full Pipeline

This script processes the expanded pain medication dataset (3,184 reviews) through:
1. Data Cleaning - Remove missing values, duplicates, clean text
2. Feature Engineering - Add date features, sentiment, word count, effectiveness bins
3. Report Statistics - Compare with original dataset

Input: 01_data/filtered/pain_meds_filtered_expanded.csv
Outputs: 
  - 01_data/cleaned/pain_meds_cleaned_expanded.csv
  - 01_data/processed/pain_meds_ml_ready_expanded.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Download VADER lexicon
print("Initializing VADER sentiment analyzer...")
try:
    nltk.download('vader_lexicon', quiet=True)
    print("✓ VADER lexicon ready")
except Exception as e:
    print(f"Warning: {e}")

# Define paths
PROJECT_DIR = Path(__file__).parent.parent.parent
INPUT_FILE = PROJECT_DIR / "01_data" / "filtered" / "pain_meds_filtered_expanded.csv"
CLEANED_OUTPUT = PROJECT_DIR / "01_data" / "cleaned" / "pain_meds_cleaned_expanded.csv"
ML_READY_OUTPUT = PROJECT_DIR / "01_data" / "processed" / "pain_meds_ml_ready_expanded.csv"

# Reference original dataset for comparison
ORIGINAL_CLEANED = PROJECT_DIR / "01_data" / "cleaned" / "pain_meds_cleaned.csv"

print("\n" + "="*80)
print("PROCESSING EXPANDED PAIN MEDICATION DATASET - FULL PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: DATA CLEANING
# ============================================================================
print("\n" + "-"*80)
print("STEP 1: DATA CLEANING")
print("-"*80)

print(f"\nLoading input file: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

initial_rows = len(df)
print(f"\nInitial dataset statistics:")
print(f"  Total rows: {initial_rows:,}")
print(f"  Columns: {list(df.columns)}")

# Check for missing values
print(f"\nChecking for missing values:")
missing_before = df.isnull().sum()
missing_total = missing_before.sum()
print(f"  Total missing values: {missing_total}")
for col in df.columns:
    if missing_before[col] > 0:
        print(f"    {col}: {missing_before[col]} ({missing_before[col]/len(df)*100:.2f}%)")

# Remove rows with missing critical columns
critical_cols = ['drugName', 'condition', 'review', 'rating', 'date']
print(f"\nRemoving rows with missing values in critical columns: {critical_cols}")
df_clean = df.dropna(subset=critical_cols)
rows_after_missing = len(df_clean)
rows_removed_missing = initial_rows - rows_after_missing
print(f"  Rows removed: {rows_removed_missing:,}")
print(f"  Rows remaining: {rows_after_missing:,}")

# Remove duplicates
print(f"\nChecking for duplicates...")
duplicates_before = df_clean.duplicated().sum()
print(f"  Duplicates found: {duplicates_before}")
if duplicates_before > 0:
    df_clean = df_clean.drop_duplicates()
    rows_after_duplicates = len(df_clean)
    print(f"  Rows removed: {duplicates_before}")
    print(f"  Rows remaining: {rows_after_duplicates:,}")

# Clean text fields
print(f"\nCleaning text fields...")
df_clean['review'] = df_clean['review'].str.strip()
df_clean['drugName'] = df_clean['drugName'].str.strip()
df_clean['condition'] = df_clean['condition'].str.strip()
print(f"✓ Text fields cleaned")

# Save cleaned dataset
print(f"\nSaving cleaned dataset to: {CLEANED_OUTPUT}")
CLEANED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(CLEANED_OUTPUT, index=False)
print(f"✓ Saved: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n" + "-"*80)
print("STEP 2: FEATURE ENGINEERING")
print("-"*80)

df_features = df_clean.copy()

# 2.1 Extract date features
print(f"\nExtracting date features from 'date' column...")
df_features['date'] = pd.to_datetime(df_features['date'], format='%d-%b-%y', errors='coerce')

df_features['year'] = df_features['date'].dt.year
df_features['month'] = df_features['date'].dt.month
df_features['day'] = df_features['date'].dt.day
df_features['day_of_week'] = df_features['date'].dt.dayofweek

print(f"✓ Added features: year, month, day, day_of_week")
print(f"  Date range: {df_features['date'].min()} to {df_features['date'].max()}")

# 2.2 Calculate review word count
print(f"\nCalculating review word count...")
df_features['review_word_count'] = df_features['review'].apply(lambda x: len(str(x).split()))
print(f"✓ Added feature: review_word_count")
print(f"  Word count range: {df_features['review_word_count'].min()} to {df_features['review_word_count'].max()}")
print(f"  Mean word count: {df_features['review_word_count'].mean():.1f}")

# 2.3 VADER Sentiment Analysis
print(f"\nPerforming VADER sentiment analysis...")
sia = SentimentIntensityAnalyzer()

sentiment_scores = df_features['review'].apply(lambda x: sia.polarity_scores(str(x)))
df_features['compound'] = sentiment_scores.apply(lambda x: x['compound'])
df_features['neg'] = sentiment_scores.apply(lambda x: x['neg'])
df_features['neu'] = sentiment_scores.apply(lambda x: x['neu'])
df_features['pos'] = sentiment_scores.apply(lambda x: x['pos'])

print(f"✓ Added sentiment features: compound, neg, neu, pos")
print(f"  Compound score range: {df_features['compound'].min():.3f} to {df_features['compound'].max():.3f}")
print(f"  Mean compound: {df_features['compound'].mean():.3f}")

# 2.4 Bin rating into effectiveness categories
print(f"\nBinning rating into effectiveness categories...")
def categorize_effectiveness(rating):
    """
    1-4: Not Effective (0)
    5-7: Partially Effective (1)
    8-10: Effective (2)
    """
    if rating <= 4:
        return 0  # Not Effective
    elif rating <= 7:
        return 1  # Partially Effective
    else:
        return 2  # Effective

df_features['effectiveness_encoded'] = df_features['rating'].apply(categorize_effectiveness)

# Add categorical label for reference
effectiveness_map = {0: 'Not Effective', 1: 'Partially Effective', 2: 'Effective'}
df_features['effectiveness'] = df_features['effectiveness_encoded'].map(effectiveness_map)

print(f"✓ Added features: effectiveness_encoded, effectiveness")
print(f"\nEffectiveness distribution:")
for code, label in effectiveness_map.items():
    count = (df_features['effectiveness_encoded'] == code).sum()
    pct = count / len(df_features) * 100
    print(f"  {label} ({code}): {count:,} ({pct:.1f}%)")

# 2.5 Create top drug and condition features
print(f"\nCreating top drug and condition features...")

# Get top 30 drugs
top_drugs = df_features['drugName'].value_counts().head(30).index.tolist()
df_features['drugName_top'] = df_features['drugName'].apply(
    lambda x: x if x in top_drugs else 'Other'
)

# Get top conditions (based on original analysis)
top_conditions = [
    'back pain', 'chronic pain', 'cluster headaches', 'headache',
    'juvenile rheumatoid arthritis', 'migraine', 'muscle pain', 
    'neck pain', 'osteoarthritis', 'rheumatoid arthritis', 
    'sciatica', 'spondyloarthritis', 'toothache'
]

# Standardize condition names (lowercase for matching)
df_features['condition_lower'] = df_features['condition'].str.lower().str.strip()
df_features['condition_top'] = df_features['condition_lower'].apply(
    lambda x: x if x in top_conditions else 'other'
)

print(f"✓ Added features: drugName_top, condition_top")
print(f"  Top drugs: {len(top_drugs)} categories + Other")
print(f"  Top conditions: {len(top_conditions)} categories + other")

# 2.6 One-hot encode drugName_top
print(f"\nOne-hot encoding drugName_top...")
drug_dummies = pd.get_dummies(df_features['drugName_top'], prefix='drug')
df_features = pd.concat([df_features, drug_dummies], axis=1)
print(f"✓ Added {len(drug_dummies.columns)} drug features")

# 2.7 One-hot encode condition_top
print(f"\nOne-hot encoding condition_top...")
condition_dummies = pd.get_dummies(df_features['condition_top'], prefix='condition')
df_features = pd.concat([df_features, condition_dummies], axis=1)
print(f"✓ Added {len(condition_dummies.columns)} condition features")

# 2.8 Additional text features
print(f"\nCalculating additional text features...")

# Review length (character count)
df_features['review_length'] = df_features['review'].apply(lambda x: len(str(x)))

# Check for positive keywords
positive_keywords = ['great', 'excellent', 'amazing', 'wonderful', 'effective', 
                    'relief', 'helped', 'best', 'good', 'works', 'perfect']
df_features['has_positive_keywords'] = df_features['review'].str.lower().apply(
    lambda x: int(any(keyword in str(x) for keyword in positive_keywords))
)

# Check for negative keywords
negative_keywords = ['terrible', 'awful', 'horrible', 'pain', 'ineffective', 
                    'useless', 'worse', 'bad', 'doesn\'t work', 'side effects']
df_features['has_negative_keywords'] = df_features['review'].str.lower().apply(
    lambda x: int(any(keyword in str(x) for keyword in negative_keywords))
)

# Average word length
df_features['avg_word_length'] = df_features['review'].apply(
    lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
)

print(f"✓ Added features: review_length, has_positive_keywords, has_negative_keywords, avg_word_length")

# Drop temporary columns
df_features = df_features.drop(columns=['condition_lower'])

# Count total features
print(f"\n✓ Total feature columns: {len(df_features.columns)}")

# Reorder columns to match original dataset structure
col_order = [
    'uniqueID', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount',
    'year', 'effectiveness', 'effectiveness_encoded', 'drugName_top', 'condition_top'
]

# Add drug columns in alphabetical order
drug_cols = sorted([c for c in df_features.columns if c.startswith('drug_')])
col_order.extend(drug_cols)

# Add condition columns in alphabetical order  
condition_cols = sorted([c for c in df_features.columns if c.startswith('condition_')])
col_order.extend(condition_cols)

# Add remaining feature columns
remaining_cols = ['review_length', 'review_word_count', 'has_positive_keywords', 
                 'has_negative_keywords', 'avg_word_length',
                 'compound', 'neg', 'neu', 'pos']
col_order.extend(remaining_cols)

# Reorder dataframe
df_features = df_features[col_order]
print(f"  Core columns: 12")
print(f"  Drug one-hot features: {len(drug_cols)}")
print(f"  Condition one-hot features: {len(condition_cols)}")
print(f"  Text/sentiment features: {len(remaining_cols)}")

# Save ML-ready dataset
print(f"\nSaving ML-ready dataset to: {ML_READY_OUTPUT}")
ML_READY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df_features.to_csv(ML_READY_OUTPUT, index=False)
print(f"✓ Saved: {df_features.shape[0]:,} rows, {df_features.shape[1]} columns")

# ============================================================================
# STEP 3: REPORT STATISTICS - COMPARE OLD VS NEW
# ============================================================================
print("\n" + "-"*80)
print("STEP 3: COMPARISON REPORT - OLD VS NEW DATASETS")
print("-"*80)

# Load original cleaned dataset for comparison
if ORIGINAL_CLEANED.exists():
    df_original = pd.read_csv(ORIGINAL_CLEANED)
    print(f"\nOriginal dataset: {ORIGINAL_CLEANED.name}")
    print(f"  Rows: {len(df_original):,}")
    print(f"  Columns: {len(df_original.columns)}")
    
    print(f"\nExpanded dataset: {CLEANED_OUTPUT.name}")
    print(f"  Rows: {len(df_clean):,}")
    print(f"  Columns: {len(df_clean.columns)}")
    
    print(f"\nIncrease:")
    print(f"  Additional rows: {len(df_clean) - len(df_original):,}")
    print(f"  Percentage increase: {(len(df_clean) - len(df_original)) / len(df_original) * 100:.1f}%")
else:
    print(f"\n⚠ Original dataset not found at {ORIGINAL_CLEANED}")
    print(f"  Skipping comparison...")

# Summary statistics
print("\n" + "="*80)
print("PROCESSING SUMMARY - EXPANDED DATASET")
print("="*80)

print(f"\nInput file: {INPUT_FILE.name}")
print(f"  Initial rows: {initial_rows:,}")

print(f"\nCleaning process:")
print(f"  Missing values removed: {rows_removed_missing:,}")
print(f"  Duplicates removed: {duplicates_before}")
print(f"  Final cleaned rows: {len(df_clean):,}")

print(f"\nFeature engineering:")
print(f"  Date features added: 4 (year, month, day, day_of_week)")
print(f"  Text features added: 5 (review_length, review_word_count, has_positive/negative_keywords, avg_word_length)")
print(f"  Sentiment features added: 4 (compound, neg, neu, pos)")
print(f"  Target features added: 2 (effectiveness_encoded, effectiveness)")
print(f"  One-hot encoded features: {len(drug_cols) + len(condition_cols)}")
print(f"    - Drug categories: {len(drug_cols)}")
print(f"    - Condition categories: {len(condition_cols)}")
print(f"  Total feature columns: {len(df_features.columns)}")

print(f"\nOutput files:")
print(f"  Cleaned: {CLEANED_OUTPUT}")
print(f"    → {len(df_clean):,} rows")
print(f"  ML-ready: {ML_READY_OUTPUT}")
print(f"    → {len(df_features):,} rows, {len(df_features.columns)} columns")

print(f"\nClass distribution (effectiveness):")
for code, label in effectiveness_map.items():
    count = (df_features['effectiveness_encoded'] == code).sum()
    pct = count / len(df_features) * 100
    print(f"  {label} ({code}): {count:,} ({pct:.1f}%)")

print("\n" + "="*80)
print("✓ PIPELINE COMPLETE - EXPANDED DATASET READY FOR MODELING")
print("="*80)
