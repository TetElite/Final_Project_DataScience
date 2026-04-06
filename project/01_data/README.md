# 📊 DATASET DIRECTORY

## Data Pipeline

```
raw/ → filtered/ → cleaned/ → processed/
280K    2,975      2,473       2,473 (65 features)
```

## Folders

### `raw/` - Original Unprocessed Data
- **drugsComTrain_raw.csv** (209,989 reviews)
- **drugsComTest_raw.csv** (70,489 reviews)
- **Total:** 280,478 drug reviews from all categories

### `filtered/` - Pain-Specific Data
- **pain_meds_filtered.csv** (2,975 reviews)
- Filtered to include only pain-related medications
- **Reduction:** 99.12% of data removed

### `cleaned/` - Cleaned & Validated
- **pain_meds_cleaned.csv** (2,473 reviews)
- Missing values removed: 0
- Duplicates removed: 502 rows
- Text cleaned (HTML entities, special chars)

### `processed/` - ML-Ready Features
- **pain_meds_ml_ready.csv** (2,473 rows × 65 columns)
- 55 features for model training
- One-hot encoded drugs (31) + conditions (13)
- Sentiment scores (4), text features (7)
- Target: effectiveness_encoded (0, 1, 2)

## Data Statistics

- **Time Range:** 2008-2017 (10 years)
- **Medications:** 49 distinct pain drugs
- **Conditions:** 13 pain-related conditions
- **Average Rating:** 7.77/10
- **Most Common:** Back pain (523), Chronic pain (518)
