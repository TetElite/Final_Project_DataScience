"""
Data loading utilities for Pain Medication Effectiveness Predictor
"""
import pandas as pd
import os

def load_raw_data():
    """Load raw dataset from Kaggle"""
    path = '../data/raw/drugsComTrain_raw.tsv'
    return pd.read_csv(path, sep='\t')

def load_filtered_data():
    """Load filtered pain medication data"""
    path = '../data/filtered/pain_meds_filtered.csv'
    return pd.read_csv(path)

def load_cleaned_data():
    """Load cleaned dataset"""
    path = '../data/cleaned/pain_meds_cleaned.csv'
    return pd.read_csv(path)

def load_ml_ready_data():
    """Load processed data ready for modeling"""
    path = '../data/processed/pain_meds_ml_ready.csv'
    return pd.read_csv(path)
