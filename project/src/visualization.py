"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution(data, column, title, figsize=(10, 6)):
    """
    Create histogram with proper styling
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the data
    column : str
        Column name to plot
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure : Created figure
    """
    plt.figure(figsize=figsize)
    data[column].hist(bins=20, edgecolor='black')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_boxplot_by_category(data, numeric_col, category_col, figsize=(12, 6)):
    """
    Create boxplot grouped by category
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the data
    numeric_col : str
        Numeric column for y-axis
    category_col : str
        Categorical column for x-axis
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure : Created figure
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=data, x=category_col, y=numeric_col)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{numeric_col} by {category_col}', fontsize=14, fontweight='bold')
    plt.xlabel(category_col, fontsize=12)
    plt.ylabel(numeric_col, fontsize=12)
    plt.tight_layout()
    return plt.gcf()

def plot_correlation_heatmap(data, figsize=(10, 8)):
    """
    Create correlation heatmap
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing numeric data
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure : Created figure
    """
    plt.figure(figsize=figsize)
    corr = data.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return plt.gcf()
