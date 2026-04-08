import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# ============================================================================
# VISUALIZATION 1: Data Cleaning Steps (Before/After Bar Chart)
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(12, 6))

# Data
steps = ['Original Data\n(Filtered)', 
         'After Removing\nMissing Values', 
         'After Removing\nDuplicates', 
         'After Text\nCleaning']
counts = [2975, 2473, 2473, 2473]
colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']

# Calculate rows removed
rows_removed = [0, 2975-2473, 0, 0]

# Create horizontal bar chart
bars = ax1.barh(steps, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels at the end of each bar
for i, (bar, count, removed) in enumerate(zip(bars, counts, rows_removed)):
    # Add count label
    ax1.text(count + 50, bar.get_y() + bar.get_height()/2, 
             f'{count:,} rows', 
             va='center', fontweight='bold', fontsize=11)
    
    # Add removed count if > 0
    if removed > 0:
        ax1.text(count + 350, bar.get_y() + bar.get_height()/2, 
                 f'(-{removed} rows)', 
                 va='center', fontsize=10, color='red', fontweight='bold')

# Customize plot
ax1.set_xlabel('Number of Records', fontsize=13, fontweight='bold')
ax1.set_ylabel('Cleaning Step', fontsize=13, fontweight='bold')
ax1.set_title('Data Cleaning Pipeline - Record Count', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlim(0, 3500)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add summary text box
summary_text = f'Total Records Removed: {2975-2473}\nRetention Rate: {(2473/2975)*100:.1f}%'
ax1.text(0.98, 0.02, summary_text, 
         transform=ax1.transAxes,
         fontsize=11,
         verticalalignment='bottom',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/macbook/CADT/Term2Year3/Data/Final_Project/project/05_analysis_results/plots/data_cleaning_steps.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved: data_cleaning_steps.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Feature Engineering Process
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(14, 10))
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title
ax2.text(5, 9.5, 'Feature Engineering Process', 
         fontsize=20, fontweight='bold', ha='center')

# Left Box - Raw Features
left_x, left_y = 0.5, 2.5
left_width, left_height = 3.5, 6

# Draw left box
left_box = FancyBboxPatch((left_x, left_y), left_width, left_height,
                          boxstyle="round,pad=0.1", 
                          edgecolor='#3498db', facecolor='#ecf0f1',
                          linewidth=3)
ax2.add_patch(left_box)

# Left box title
ax2.text(left_x + left_width/2, left_y + left_height - 0.3, 
         'Raw Features\n(6 columns)', 
         fontsize=14, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.7, edgecolor='black'))

# Raw features list
raw_features = [
    '1. drugName',
    '2. condition', 
    '3. review',
    '4. rating',
    '5. date',
    '6. usefulCount'
]

y_pos = left_y + left_height - 1.5
for feature in raw_features:
    ax2.text(left_x + 0.3, y_pos, feature, 
             fontsize=12, va='top', family='monospace')
    y_pos -= 0.7

# Arrow in the middle
arrow = FancyArrowPatch((left_x + left_width + 0.2, 5.5), 
                       (left_x + left_width + 1.8, 5.5),
                       arrowstyle='->', 
                       mutation_scale=40, 
                       linewidth=4,
                       color='#e74c3c')
ax2.add_patch(arrow)

# Arrow label
ax2.text(left_x + left_width + 1, 6, 'Feature\nEngineering', 
         fontsize=13, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.7, edgecolor='black'))

# Right Box - Engineered Features
right_x, right_y = 6, 2.5
right_width, right_height = 3.5, 6

# Draw right box
right_box = FancyBboxPatch((right_x, right_y), right_width, right_height,
                           boxstyle="round,pad=0.1", 
                           edgecolor='#2ecc71', facecolor='#ecf0f1',
                           linewidth=3)
ax2.add_patch(right_box)

# Right box title
ax2.text(right_x + right_width/2, right_y + right_height - 0.3, 
         'Engineered Features\n(67 columns)', 
         fontsize=14, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.7, edgecolor='black'))

# Engineered features list
engineered_features = [
    'Original Features (6)',
    '  • All raw columns',
    '',
    'Sentiment Features (4)',
    '  • compound, neg, neu, pos',
    '',
    'Date Features (4)',
    '  • year, month, day, day_of_week',
    '',
    'Text Features (2)',
    '  • review_length',
    '  • review_word_count',
    '',
    'One-Hot Encoded (46)',
    '  • drug categories',
    '  • condition categories',
    '',
    'Target Variable (1)',
    '  • effectiveness'
]

y_pos = right_y + right_height - 1.2
line_spacing = 0.3
for feature in engineered_features:
    if feature == '':
        y_pos -= 0.15
        continue
    
    # Bold for category headers
    if feature.endswith(')') and not feature.startswith('  •'):
        ax2.text(right_x + 0.2, y_pos, feature, 
                fontsize=10, va='top', fontweight='bold')
    else:
        ax2.text(right_x + 0.2, y_pos, feature, 
                fontsize=9, va='top')
    y_pos -= line_spacing

# Add statistics boxes at the bottom
stats_y = 1.5

# Original data stat
ax2.text(2.25, stats_y, '2,975\nOriginal Records', 
         fontsize=11, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.6, edgecolor='black'))

# Arrow
arrow2 = FancyArrowPatch((3.2, stats_y + 0.1), (4.3, stats_y + 0.1),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax2.add_patch(arrow2)

# Middle stat
ax2.text(5, stats_y, '2,473\nCleaned Records', 
         fontsize=11, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.6, edgecolor='black'))

# Arrow
arrow3 = FancyArrowPatch((5.8, stats_y + 0.1), (6.9, stats_y + 0.1),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax2.add_patch(arrow3)

# Final stat
ax2.text(7.75, stats_y, '2,473 × 67\nFinal Dataset', 
         fontsize=11, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.6, edgecolor='black'))

# Add footer
footer_text = 'Feature engineering expanded the dataset from 6 to 67 columns while maintaining data quality'
ax2.text(5, 0.5, footer_text, 
         fontsize=10, ha='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/macbook/CADT/Term2Year3/Data/Final_Project/project/05_analysis_results/plots/feature_engineering_process.png', 
            dpi=300, bbox_inches='tight')
print("✓ Saved: feature_engineering_process.png")
plt.close()

print("\n" + "="*60)
print("VISUALIZATION GENERATION COMPLETE")
print("="*60)
print("\nFiles saved to:")
print("1. project/05_analysis_results/plots/data_cleaning_steps.png")
print("2. project/05_analysis_results/plots/feature_engineering_process.png")
print("\nBoth visualizations are presentation-ready at 300 DPI.")
