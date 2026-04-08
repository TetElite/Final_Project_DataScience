#!/usr/bin/env python3
"""
Create a professional dataset preview table visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the cleaned dataset
df = pd.read_csv('/Users/macbook/CADT/Term2Year3/Data/Final_Project/project/01_data/cleaned/pain_meds_cleaned.csv')

# Select first 5 rows and specific columns
columns_to_show = ['drugName', 'condition', 'review', 'rating', 'date', 'usefulCount']
preview_df = df[columns_to_show].head(5).copy()

# Truncate review text to 50 characters
preview_df['review'] = preview_df['review'].apply(lambda x: str(x)[:50] + '...' if len(str(x)) > 50 else str(x))

# Prepare data for table
table_data = []
for idx, row in preview_df.iterrows():
    table_data.append([
        row['drugName'],
        row['condition'],
        row['review'],
        row['rating'],
        row['date'],
        row['usefulCount']
    ])

# Column headers
headers = ['Drug Name', 'Condition', 'Review', 'Rating', 'Date', 'Useful Count']

# Create figure and axis
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=table_data,
                colLabels=headers,
                cellLoc='left',
                loc='center',
                colWidths=[0.15, 0.15, 0.35, 0.08, 0.12, 0.12])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header row
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(weight='bold', color='white', fontsize=10)
    cell.set_height(0.08)

# Style data rows with alternating colors
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')  # Light gray
        else:
            cell.set_facecolor('white')
        cell.set_edgecolor('#bdc3c7')
        cell.set_height(0.08)

# Add title
plt.title('Dataset Preview - Pain Medication Reviews', 
          fontsize=16, 
          fontweight='bold', 
          pad=20,
          color='#2c3e50')

# Add subtitle with dataset info
fig.text(0.5, 0.05, f'Showing 5 of {len(df):,} total records', 
         ha='center', 
         fontsize=10, 
         style='italic',
         color='#7f8c8d')

# Save with high resolution
plt.tight_layout()
plt.savefig('/Users/macbook/CADT/Term2Year3/Data/Final_Project/project/05_analysis_results/plots/dataset_preview.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

print("✓ Dataset preview table created successfully!")
print(f"  File: project/05_analysis_results/plots/dataset_preview.png")
print(f"  Resolution: 300 DPI")
print(f"  Rows shown: 5 of {len(df):,} total records")
