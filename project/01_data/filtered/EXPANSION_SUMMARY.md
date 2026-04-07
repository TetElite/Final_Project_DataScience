# Pain Medication Dataset Expansion Summary

**Date:** April 7, 2026  
**Script:** `expand_pain_meds.py`  
**Output File:** `pain_meds_filtered_expanded.csv`

---

## Executive Summary

The pain medication dataset has been successfully expanded to include **nerve pain medications**, resulting in a **28.8% increase** in available data.

### Key Results:
- **Old Count:** 2,473 reviews
- **New Count:** 3,184 reviews  
- **Increase:** +711 reviews (28.8%)
- **New Medications Added:** 9 terms (6 nerve pain + 3 paracetamol brands*)

*Note: Paracetamol brands (panadol, paracetamol, calpol) found 0 reviews in this dataset.

---

## Medication List Expansion

### Original Medications (15 terms)
✓ ibuprofen, acetaminophen, naproxen, aspirin, diclofenac  
✓ tramadol, hydrocodone, oxycodone  
✓ tylenol, advil, aleve, motrin  
✓ celebrex, meloxicam, indomethacin

### NEW: Nerve Pain Medications (6 terms)
✅ **gabapentin** - 160 reviews  
✅ **pregabalin** - 142 reviews  
✅ **lyrica** - 123 reviews (brand name for pregabalin)  
✅ **duloxetine** - 141 reviews  
✅ **cymbalta** - 129 reviews (brand name for duloxetine)  
✅ **neurontin** - 16 reviews (brand name for gabapentin)

### NEW: Paracetamol Brand Names (3 terms)
❌ panadol - 0 reviews  
❌ paracetamol - 0 reviews  
❌ calpol - 0 reviews

**Total Medication Terms:** 24

---

## Pain Conditions (14 - unchanged)

headache, migraine, back pain, arthritis, sciatica, fibromyalgia, toothache, neck pain, joint pain, osteoarthritis, rheumatoid arthritis, chronic pain, neuropathic pain, muscle pain

---

## Nerve Pain Medication Impact

### Coverage by Medication

| Medication | Reviews | Top Condition | Avg Rating |
|-----------|---------|---------------|------------|
| **Gabapentin** | 160 | Neuropathic Pain (84) | 7.1/10 |
| **Pregabalin** | 142 | Neuropathic Pain (131) | 6.7/10 |
| **Lyrica** | 123 | Neuropathic Pain (118) | 6.6/10 |
| **Duloxetine** | 141 | Chronic Pain (95) | 6.5/10 |
| **Cymbalta** | 129 | Chronic Pain (86) | 6.5/10 |
| **Neurontin** | 16 | Migraine (16) | 8.4/10 |
| **Total** | **711** | - | **6.8/10** |

### Neuropathic Pain Reviews

**Total Neuropathic Pain Reviews:** 333

**Top Medications for Neuropathic Pain:**
1. Pregabalin - 131 reviews (39.3%)
2. Lyrica - 118 reviews (35.4%)
3. Gabapentin - 84 reviews (25.2%)

These three medications account for **100%** of neuropathic pain medication reviews in the expanded dataset.

---

## Expanded Dataset Statistics

### Overall Metrics
- **Total Records:** 3,184
- **Unique Drugs:** 55
- **Unique Conditions:** 15

### Rating Distribution
| Rating | Count | Percentage |
|--------|-------|------------|
| 10 | 1,117 | 35.1% |
| 9 | 658 | 20.7% |
| 8 | 418 | 13.1% |
| 7 | 206 | 6.5% |
| 1 | 352 | 11.1% |
| Other (2-6) | 433 | 13.6% |

### Top 10 Medications
1. Diclofenac - 290 reviews
2. Tramadol - 288 reviews
3. Oxycodone - 245 reviews
4. Naproxen - 207 reviews
5. Acetaminophen / hydrocodone - 205 reviews
6. Acetaminophen / butalbital / caffeine - 176 reviews
7. **Gabapentin - 160 reviews** ⭐ NEW
8. Meloxicam - 152 reviews
9. **Pregabalin - 142 reviews** ⭐ NEW
10. **Duloxetine - 141 reviews** ⭐ NEW

*Note: 3 of top 10 medications are newly added nerve pain medications!*

### Top 10 Conditions
1. Chronic Pain - 699 reviews (22.0%)
2. Back Pain - 570 reviews (17.9%)
3. Osteoarthritis - 458 reviews (14.4%)
4. Headache - 403 reviews (12.7%)
5. Migraine - 374 reviews (11.7%)
6. **Neuropathic Pain - 333 reviews (10.5%)** ⭐ INCREASED
7. Rheumatoid Arthritis - 167 reviews (5.2%)
8. Sciatica - 76 reviews (2.4%)
9. Muscle Pain - 64 reviews (2.0%)
10. Migraine Prevention - 16 reviews (0.5%)

---

## Impact Analysis

### Benefits of Expansion

1. **Increased Dataset Size:** 28.8% more training data
2. **Better Neuropathic Pain Coverage:** From limited to comprehensive
3. **Nerve Pain Medication Representation:** Now includes gabapentinoids and SNRIs
4. **More Diverse Pain Types:** Better coverage of chronic and neuropathic pain
5. **Improved Model Training:** More examples for rare pain conditions

### Data Quality

- **Filtering Method:** AND logic (condition AND medication must match)
- **Data Retention:** 1.48% of raw data (high quality, focused dataset)
- **Average Rating:** Balanced distribution with slight positive skew
- **Review Quality:** Comprehensive reviews with detailed patient experiences

---

## Files Generated

| File | Size | Records | Description |
|------|------|---------|-------------|
| `pain_meds_filtered.csv` | 964 KB | 2,473 | Original filtered dataset |
| `pain_meds_filtered_expanded.csv` | 1.3 MB | 3,184 | Expanded dataset with nerve pain meds |
| `expand_pain_meds.py` | - | - | Filtering script |
| `EXPANSION_SUMMARY.md` | - | - | This summary document |

---

## Next Steps

1. **Data Cleaning:** Process expanded dataset through cleaning pipeline
2. **Feature Engineering:** Generate features for nerve pain medications
3. **Model Retraining:** Retrain models on expanded dataset
4. **Performance Evaluation:** Compare model performance on expanded data
5. **Analysis:** Analyze nerve pain medication effectiveness patterns

---

## Script Execution Details

**Execution Date:** 2026-04-07 15:14:16  
**Script Location:** `project/03_source_code/data_processing/expand_pain_meds.py`  
**Output Location:** `project/01_data/filtered/pain_meds_filtered_expanded.csv`  
**Execution Time:** ~3 seconds  
**Python Version:** 3.x  
**Dependencies:** pandas, numpy, os, datetime

---

## Conclusion

The dataset expansion was **highly successful**, adding 711 new reviews focused on nerve pain medications. This represents a **28.8% increase** in available data and significantly improves coverage for:

- **Neuropathic pain** conditions
- **Nerve pain medications** (gabapentinoids and SNRIs)
- **Chronic pain** management approaches
- **Alternative pain medication** strategies

The expanded dataset provides a more comprehensive foundation for training pain medication effectiveness prediction models.

---

**Generated by:** `expand_pain_meds.py`  
**Last Updated:** April 7, 2026
