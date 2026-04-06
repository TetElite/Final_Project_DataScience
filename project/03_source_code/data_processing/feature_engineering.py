"""
Data processing script to add VADER sentiment features to ML-ready dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if needed
print("Downloading NLTK VADER lexicon...")
try:
    nltk.download('vader_lexicon', quiet=True)
    print("✓ VADER lexicon ready")
except Exception as e:
    print(f"Warning: NLTK download issue: {e}")

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "pain_meds_ml_ready.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "pain_meds_ml_ready.csv"

print("\n" + "="*60)
print("ADDING VADER SENTIMENT FEATURES")
print("="*60)

# Load data
print(f"\nLoading data from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
print(f"✓ Data loaded: {df.shape}")
print(f"✓ Original columns: {len(df.columns)}")

# Check if sentiment features already exist
sentiment_cols = ['compound', 'neg', 'neu', 'pos']
if all(col in df.columns for col in sentiment_cols):
    print("\n⚠ Sentiment features already exist in the dataset!")
    print("  Columns found:", sentiment_cols)
    print("  Recalculating sentiment scores...")
    # Drop existing sentiment columns to recalculate
    df = df.drop(columns=sentiment_cols)

# Initialize sentiment analyzer
print("\nInitializing VADER sentiment analyzer...")
sia = SentimentIntensityAnalyzer()
print("✓ Sentiment analyzer ready")

# Calculate VADER sentiment scores for each review
print("\nCalculating VADER sentiment scores for all reviews...")
print(f"  Processing {len(df)} reviews...")

# Apply sentiment analysis
sentiment_scores = df['review'].apply(lambda x: sia.polarity_scores(str(x)))

# Extract individual scores
df['compound'] = sentiment_scores.apply(lambda x: x['compound'])
df['neg'] = sentiment_scores.apply(lambda x: x['neg'])
df['neu'] = sentiment_scores.apply(lambda x: x['neu'])
df['pos'] = sentiment_scores.apply(lambda x: x['pos'])

print(f"✓ Added sentiment features: compound, neg, neu, pos")
print(f"\nSentiment Statistics:")
print(f"  Compound score range: {df['compound'].min():.3f} to {df['compound'].max():.3f}")
print(f"  Mean compound: {df['compound'].mean():.3f}")
print(f"  Median compound: {df['compound'].median():.3f}")
print(f"  Negative score mean: {df['neg'].mean():.3f}")
print(f"  Neutral score mean: {df['neu'].mean():.3f}")
print(f"  Positive score mean: {df['pos'].mean():.3f}")

# Show correlation with effectiveness
print(f"\nCorrelation with effectiveness_encoded:")
print(f"  compound: {df['compound'].corr(df['effectiveness_encoded']):.3f}")
print(f"  pos: {df['pos'].corr(df['effectiveness_encoded']):.3f}")
print(f"  neg: {df['neg'].corr(df['effectiveness_encoded']):.3f}")

# Show distribution by effectiveness
print(f"\nMean compound score by effectiveness:")
for eff_val in sorted(df['effectiveness_encoded'].unique()):
    mean_compound = df[df['effectiveness_encoded'] == eff_val]['compound'].mean()
    eff_name = ['Not Effective', 'Partially Effective', 'Effective'][eff_val]
    print(f"  {eff_name} ({eff_val}): {mean_compound:.3f}")

# Save updated dataset
print(f"\n" + "-"*60)
print(f"Saving updated dataset to: {OUTPUT_PATH}")
df.to_csv(OUTPUT_PATH, index=False)
print(f"✓ Saved: {df.shape}")
print(f"✓ Total columns now: {len(df.columns)}")

# Verify sentiment columns are in the saved file
saved_df = pd.read_csv(OUTPUT_PATH)
has_all_sentiment = all(col in saved_df.columns for col in sentiment_cols)
print(f"\n✓ Verification: Sentiment columns in saved file: {has_all_sentiment}")
if has_all_sentiment:
    print(f"  Columns confirmed: {sentiment_cols}")
else:
    missing = [col for col in sentiment_cols if col not in saved_df.columns]
    print(f"  ✗ Missing columns: {missing}")

print("\n" + "="*60)
print("SENTIMENT FEATURE EXTRACTION COMPLETE")
print("="*60)
print(f"\nDataset ready for model training with sentiment features!")
print(f"Next step: Run retrain_model.py to train with sentiment features")
