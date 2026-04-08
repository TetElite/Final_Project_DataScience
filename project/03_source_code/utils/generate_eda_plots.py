"""
Generate EDA Visualizations for Presentation
Extracts and saves all key plots as PNG files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "01_data" / "cleaned" / "pain_meds_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "05_analysis_results" / "plots"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df)} reviews\n")

# 1. RATING DISTRIBUTION
print("1. Creating rating distribution...")
plt.figure(figsize=(10, 6))
plt.hist(df['rating'], bins=10, edgecolor='black', color='skyblue', alpha=0.7)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Pain Medication Ratings', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rating_distribution.png", bbox_inches='tight', dpi=300)
plt.close()
print("   ✓ rating_distribution.png")

# 2. TOP CONDITIONS
print("2. Creating top conditions chart...")
plt.figure(figsize=(12, 6))
top_conditions = df['condition'].value_counts().head(10)
top_conditions.plot(kind='barh', color='steelblue', edgecolor='black')
plt.xlabel('Number of Reviews', fontsize=12)
plt.ylabel('Condition', fontsize=12)
plt.title('Top 10 Pain Conditions', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_conditions.png", bbox_inches='tight', dpi=300)
plt.close()
print("   ✓ top_conditions.png")

# 3. TOP MEDICATIONS
print("3. Creating top medications chart...")
plt.figure(figsize=(12, 6))
top_drugs = df['drugName'].value_counts().head(10)
top_drugs.plot(kind='barh', color='coral', edgecolor='black')
plt.xlabel('Number of Reviews', fontsize=12)
plt.ylabel('Medication', fontsize=12)
plt.title('Top 10 Pain Medications', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_medications.png", bbox_inches='tight', dpi=300)
plt.close()
print("   ✓ top_medications.png")

# 4. BOXPLOT BY CONDITION
print("4. Creating boxplot by condition...")
plt.figure(figsize=(12, 6))
top_5 = df['condition'].value_counts().head(5).index
df_filtered = df[df['condition'].isin(top_5)]
sns.boxplot(data=df_filtered, x='condition', y='rating', palette='Set2')
plt.xlabel('Condition', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.title('Rating Distribution by Top 5 Conditions', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ratings_by_condition_boxplot.png", bbox_inches='tight', dpi=300)
plt.close()
print("   ✓ ratings_by_condition_boxplot.png")

# 5. CORRELATION HEATMAP
print("5. Creating correlation heatmap...")
plt.figure(figsize=(10, 8))
numeric_data = df[['rating', 'usefulCount', 'year']].corr()
sns.heatmap(numeric_data, annot=True, cmap='coolwarm', center=0, fmt='.2f',
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", bbox_inches='tight', dpi=300)
plt.close()
print("   ✓ correlation_heatmap.png")

# SUMMARY
print("\n" + "="*60)
print("✅ ALL 5 VISUALIZATIONS GENERATED!")
print("="*60)
print(f"\nLocation: {OUTPUT_DIR}\n")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    size = file.stat().st_size / 1024
    print(f"  {file.name:40s} ({size:6.1f} KB)")

print("\n📊 Ready for your presentation!")
