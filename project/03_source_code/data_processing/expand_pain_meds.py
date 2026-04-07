"""
Expand Pain Medication Dataset to Include Nerve Pain Medications
=================================================================

This script expands the pain medication dataset by adding nerve pain medications
and additional paracetamol brand names to improve coverage.

EXPANDED MEDICATION LIST (23 terms):
------------------------------------
Original medications (15): 
    ibuprofen, acetaminophen, naproxen, aspirin, diclofenac, tramadol, 
    hydrocodone, oxycodone, tylenol, advil, aleve, motrin, celebrex, 
    meloxicam, indomethacin

NEW Nerve Pain Medications (6):
    gabapentin, pregabalin, lyrica, neurontin, duloxetine, cymbalta

NEW Paracetamol Brand Names (3):
    panadol, paracetamol, calpol
    
PAIN CONDITIONS (14 - unchanged):
---------------------------------
    headache, migraine, back pain, arthritis, sciatica, fibromyalgia, 
    toothache, neck pain, joint pain, osteoarthritis, rheumatoid arthritis, 
    chronic pain, neuropathic pain, muscle pain
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_raw_data():
    """Load raw training and test datasets"""
    print("="*70)
    print("LOADING RAW DATA")
    print("="*70)
    
    train_path = '../../01_data/raw/drugsComTrain_raw.csv'
    test_path = '../../01_data/raw/drugsComTest_raw.csv'
    
    print(f"\nLoading training dataset from: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"✓ Training dataset loaded: {len(df_train):,} records")
    
    print(f"\nLoading test dataset from: {test_path}")
    df_test = pd.read_csv(test_path)
    print(f"✓ Test dataset loaded: {len(df_test):,} records")
    
    print(f"\nTotal raw records: {len(df_train) + len(df_test):,}")
    
    return df_train, df_test

def define_filters():
    """Define pain conditions and EXPANDED medication list"""
    
    # Pain conditions (unchanged - 14 conditions)
    pain_conditions = [
        'headache', 'migraine', 'back pain', 'arthritis', 'sciatica', 
        'fibromyalgia', 'toothache', 'neck pain', 'joint pain', 
        'osteoarthritis', 'rheumatoid arthritis', 'chronic pain', 
        'neuropathic pain', 'muscle pain'
    ]
    
    # EXPANDED medication list (23 total)
    # Original 15 medications
    original_medications = [
        'ibuprofen', 'acetaminophen', 'naproxen', 'aspirin', 'diclofenac', 
        'tramadol', 'hydrocodone', 'oxycodone', 'tylenol', 'advil', 
        'aleve', 'motrin', 'celebrex', 'meloxicam', 'indomethacin'
    ]
    
    # NEW: Nerve pain medications (6)
    nerve_pain_medications = [
        'gabapentin', 'pregabalin', 'lyrica', 'neurontin', 
        'duloxetine', 'cymbalta'
    ]
    
    # NEW: Additional paracetamol brand names (3)
    paracetamol_brands = [
        'panadol', 'paracetamol', 'calpol'
    ]
    
    # Combine all medications
    pain_medications = original_medications + nerve_pain_medications + paracetamol_brands
    
    return pain_conditions, pain_medications, original_medications, nerve_pain_medications, paracetamol_brands

def filter_data(df_train, df_test, pain_conditions, pain_medications):
    """Filter datasets using AND logic (condition AND medication)"""
    
    print("\n" + "="*70)
    print("FILTERING DATA WITH EXPANDED MEDICATION LIST")
    print("="*70)
    
    # Create regex patterns
    condition_pattern = '|'.join(pain_conditions)
    medication_pattern = '|'.join(pain_medications)
    
    print(f"\nConditions to match: {len(pain_conditions)}")
    print(f"Medications to match: {len(pain_medications)}")
    
    # Filter train dataset - BOTH condition AND medication must match
    print("\nFiltering training dataset...")
    train_condition_mask = df_train['condition'].str.contains(condition_pattern, case=False, na=False, regex=True)
    train_drug_mask = df_train['drugName'].str.contains(medication_pattern, case=False, na=False, regex=True)
    filtered_train = df_train[train_condition_mask & train_drug_mask].copy()
    
    print(f"✓ Training: {len(df_train):,} → {len(filtered_train):,} records "
          f"({(len(filtered_train)/len(df_train))*100:.2f}% retained)")
    
    # Filter test dataset - BOTH condition AND medication must match
    print("\nFiltering test dataset...")
    test_condition_mask = df_test['condition'].str.contains(condition_pattern, case=False, na=False, regex=True)
    test_drug_mask = df_test['drugName'].str.contains(medication_pattern, case=False, na=False, regex=True)
    filtered_test = df_test[test_condition_mask & test_drug_mask].copy()
    
    print(f"✓ Test: {len(df_test):,} → {len(filtered_test):,} records "
          f"({(len(filtered_test)/len(df_test))*100:.2f}% retained)")
    
    # Combine datasets
    print("\nCombining filtered datasets...")
    filtered_df = pd.concat([filtered_train, filtered_test], ignore_index=True)
    
    print(f"\n{'='*70}")
    print(f"TOTAL FILTERED RECORDS: {len(filtered_df):,}")
    print(f"{'='*70}")
    
    return filtered_df

def analyze_new_coverage(filtered_df, nerve_pain_meds, paracetamol_brands):
    """Analyze coverage from newly added medications"""
    
    print("\n" + "="*70)
    print("ANALYZING NEW MEDICATION COVERAGE")
    print("="*70)
    
    # Count reviews for nerve pain medications
    nerve_pain_pattern = '|'.join(nerve_pain_meds)
    nerve_pain_mask = filtered_df['drugName'].str.contains(nerve_pain_pattern, case=False, na=False, regex=True)
    nerve_pain_count = nerve_pain_mask.sum()
    
    print(f"\nNerve Pain Medications ({len(nerve_pain_meds)} terms):")
    print(f"  Total reviews: {nerve_pain_count:,}")
    
    # Breakdown by specific nerve pain medication
    print("\n  Breakdown by medication:")
    for med in nerve_pain_meds:
        count = filtered_df['drugName'].str.contains(med, case=False, na=False, regex=True).sum()
        if count > 0:
            print(f"    - {med.capitalize()}: {count:,} reviews")
    
    # Count reviews for paracetamol brands
    paracetamol_pattern = '|'.join(paracetamol_brands)
    paracetamol_mask = filtered_df['drugName'].str.contains(paracetamol_pattern, case=False, na=False, regex=True)
    paracetamol_count = paracetamol_mask.sum()
    
    print(f"\nParacetamol Brand Names ({len(paracetamol_brands)} terms):")
    print(f"  Total reviews: {paracetamol_count:,}")
    
    # Breakdown by specific paracetamol brand
    print("\n  Breakdown by brand:")
    for brand in paracetamol_brands:
        count = filtered_df['drugName'].str.contains(brand, case=False, na=False, regex=True).sum()
        if count > 0:
            print(f"    - {brand.capitalize()}: {count:,} reviews")
    
    total_new = nerve_pain_count + paracetamol_count
    print(f"\n{'='*70}")
    print(f"TOTAL NEW REVIEWS ADDED: {total_new:,}")
    print(f"  - From nerve pain meds: {nerve_pain_count:,}")
    print(f"  - From paracetamol brands: {paracetamol_count:,}")
    print(f"{'='*70}")
    
    return nerve_pain_count, paracetamol_count

def compare_with_old_dataset(new_count):
    """Compare with original dataset count"""
    
    print("\n" + "="*70)
    print("COMPARISON WITH ORIGINAL DATASET")
    print("="*70)
    
    old_count = 2473  # Original count from the project
    increase = new_count - old_count
    percent_increase = (increase / old_count) * 100
    
    print(f"\nOriginal dataset count: {old_count:,} reviews")
    print(f"Expanded dataset count: {new_count:,} reviews")
    print(f"Increase: {increase:,} reviews ({percent_increase:.1f}%)")
    
    return old_count, increase, percent_increase

def save_filtered_data(filtered_df):
    """Save filtered dataset to CSV"""
    
    output_dir = '../../01_data/filtered'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'pain_meds_filtered_expanded.csv')
    
    print("\n" + "="*70)
    print("SAVING FILTERED DATA")
    print("="*70)
    
    print(f"\nSaving to: {output_path}")
    filtered_df.to_csv(output_path, index=False)
    print(f"✓ File saved successfully!")
    print(f"  Records saved: {len(filtered_df):,}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return output_path

def display_summary_statistics(filtered_df):
    """Display summary statistics of the filtered dataset"""
    
    print("\n" + "="*70)
    print("DATASET SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal records: {len(filtered_df):,}")
    print(f"Unique drugs: {filtered_df['drugName'].nunique():,}")
    print(f"Unique conditions: {filtered_df['condition'].nunique():,}")
    
    print(f"\nRating distribution:")
    rating_dist = filtered_df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        print(f"  Rating {int(rating)}: {count:,} reviews ({count/len(filtered_df)*100:.1f}%)")
    
    print(f"\nTop 10 Medications:")
    top_drugs = filtered_df['drugName'].value_counts().head(10)
    for i, (drug, count) in enumerate(top_drugs.items(), 1):
        print(f"  {i:2}. {drug}: {count:,} reviews")
    
    print(f"\nTop 10 Conditions:")
    top_conditions = filtered_df['condition'].value_counts().head(10)
    for i, (condition, count) in enumerate(top_conditions.items(), 1):
        print(f"  {i:2}. {condition}: {count:,} reviews")

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("PAIN MEDICATION DATASET EXPANSION")
    print("Adding Nerve Pain Medications + Paracetamol Brand Names")
    print("="*70)
    print(f"\nExecution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load raw data
    df_train, df_test = load_raw_data()
    
    # Step 2: Define filters
    pain_conditions, pain_medications, original_meds, nerve_pain_meds, paracetamol_brands = define_filters()
    
    print("\n" + "="*70)
    print("MEDICATION LIST BREAKDOWN")
    print("="*70)
    print(f"\nOriginal medications ({len(original_meds)}):")
    print(f"  {', '.join(original_meds)}")
    print(f"\nNEW Nerve pain medications ({len(nerve_pain_meds)}):")
    print(f"  {', '.join(nerve_pain_meds)}")
    print(f"\nNEW Paracetamol brands ({len(paracetamol_brands)}):")
    print(f"  {', '.join(paracetamol_brands)}")
    print(f"\nTOTAL MEDICATIONS: {len(pain_medications)}")
    
    # Step 3: Filter data
    filtered_df = filter_data(df_train, df_test, pain_conditions, pain_medications)
    
    # Step 4: Analyze new coverage
    nerve_count, paracetamol_count = analyze_new_coverage(filtered_df, nerve_pain_meds, paracetamol_brands)
    
    # Step 5: Compare with original dataset
    old_count, increase, percent_increase = compare_with_old_dataset(len(filtered_df))
    
    # Step 6: Display summary statistics
    display_summary_statistics(filtered_df)
    
    # Step 7: Save filtered data
    output_path = save_filtered_data(filtered_df)
    
    # Final summary
    print("\n" + "="*70)
    print("EXECUTION COMPLETE!")
    print("="*70)
    print(f"\n✓ Expanded dataset created successfully")
    print(f"✓ Output file: {output_path}")
    print(f"✓ Old count: {old_count:,} reviews")
    print(f"✓ New count: {len(filtered_df):,} reviews")
    print(f"✓ Increase: +{increase:,} reviews ({percent_increase:.1f}%)")
    print(f"✓ New medications added: {len(nerve_pain_meds) + len(paracetamol_brands)}")
    print(f"\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
