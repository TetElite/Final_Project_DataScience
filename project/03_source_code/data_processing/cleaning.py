"""
Data cleaning utilities
"""
import pandas as pd
import numpy as np

def standardize_text(text_series):
    """
    Lowercase, strip spaces, standardize variations
    
    Parameters:
    -----------
    text_series : pd.Series
        Series containing text to standardize
        
    Returns:
    --------
    pd.Series : Standardized text
    """
    return text_series.str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)

def handle_missing_values(df, numeric_strategy='median', categorical_strategy='mode'):
    """
    Handle missing values with specified strategy
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with potential missing values
    numeric_strategy : str
        Strategy for numeric columns ('mean', 'median', 'drop')
    categorical_strategy : str
        Strategy for categorical columns ('mode', 'drop')
        
    Returns:
    --------
    pd.DataFrame : DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    # Numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            if numeric_strategy == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif numeric_strategy == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    # Categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            if categorical_strategy == 'mode':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    return df_clean

def detect_outliers_iqr(series, multiplier=1.5):
    """
    Detect outliers using IQR method
    
    Parameters:
    -----------
    series : pd.Series
        Numeric series to check for outliers
    multiplier : float
        IQR multiplier (default 1.5)
        
    Returns:
    --------
    pd.Series : Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers
