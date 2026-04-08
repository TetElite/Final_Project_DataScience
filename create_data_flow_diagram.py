import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(12, 14), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

# Define colors
box_color = '#E8F4F8'
border_color = '#2C5F7C'
arrow_color = '#2C5F7C'
text_color = '#1A1A1A'
label_color = '#555555'

# Define pipeline steps with positions
steps = [
    {
        'title': 'Raw Data',
        'details': '215,063 reviews',
        'y': 14.5,
        'type': 'start'
    },
    {
        'title': 'Filtering',
        'details': 'Pain medications + conditions',
        'y': 12.5,
        'type': 'process'
    },
    {
        'title': 'Filtered Data',
        'details': '3,184 reviews',
        'y': 10.5,
        'type': 'data'
    },
    {
        'title': 'Data Cleaning',
        'details': 'Remove missing/duplicates',
        'y': 8.5,
        'type': 'process'
    },
    {
        'title': 'Cleaned Data',
        'details': '3,184 reviews',
        'y': 6.5,
        'type': 'data'
    },
    {
        'title': 'Feature Engineering',
        'details': 'Sentiment, Date features, Encoding',
        'y': 4.5,
        'type': 'process'
    },
    {
        'title': 'ML-Ready Dataset',
        'details': '3,184 × 67 features',
        'y': 2.5,
        'type': 'data'
    },
    {
        'title': 'Model Training',
        'details': 'Random Forest + SMOTE',
        'y': 0.5,
        'type': 'process'
    }
]

# Final result box
result = {
    'title': 'Trained Model',
    'details': '70.49% accuracy',
    'y': -1.5,
    'type': 'result'
}

def create_box(ax, x, y, width, height, title, details, box_type):
    """Create a styled box for the flowchart"""
    
    # Different styling based on type
    if box_type == 'start':
        facecolor = '#D4E6F1'
        edgecolor = '#2874A6'
        linewidth = 3
    elif box_type == 'result':
        facecolor = '#D5F4E6'
        edgecolor = '#229954'
        linewidth = 3
    elif box_type == 'process':
        facecolor = '#FEF5E7'
        edgecolor = '#D68910'
        linewidth = 2
    else:  # data
        facecolor = '#E8F4F8'
        edgecolor = '#2C5F7C'
        linewidth = 2
    
    # Create rounded rectangle
    fancy_box = FancyBboxPatch(
        (x - width/2, y - height/2),
        width, height,
        boxstyle="round,pad=0.1",
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        zorder=2
    )
    ax.add_patch(fancy_box)
    
    # Add title text
    ax.text(x, y + 0.15, title,
            ha='center', va='center',
            fontsize=13, fontweight='bold',
            color=text_color,
            zorder=3)
    
    # Add details text
    ax.text(x, y - 0.15, details,
            ha='center', va='center',
            fontsize=10,
            color=label_color,
            style='italic',
            zorder=3)

def create_arrow(ax, x, y_start, y_end):
    """Create a styled arrow between boxes"""
    arrow = FancyArrowPatch(
        (x, y_start), (x, y_end),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color=arrow_color,
        linewidth=2.5,
        zorder=1
    )
    ax.add_patch(arrow)

# Draw all steps
x_center = 5
box_width = 4
box_height = 0.8

for step in steps:
    create_box(ax, x_center, step['y'], box_width, box_height,
               step['title'], step['details'], step['type'])

# Draw result box
create_box(ax, x_center, result['y'], box_width, box_height,
           result['title'], result['details'], result['type'])

# Draw arrows between all boxes
arrow_gap = 0.5
for i in range(len(steps) - 1):
    y_start = steps[i]['y'] - box_height/2 - 0.05
    y_end = steps[i+1]['y'] + box_height/2 + 0.05
    create_arrow(ax, x_center, y_start, y_end)

# Draw final arrow to result
y_start = steps[-1]['y'] - box_height/2 - 0.05
y_end = result['y'] + box_height/2 + 0.05
create_arrow(ax, x_center, y_start, y_end)

# Add title
ax.text(5, 15.8, 'Data Processing Pipeline',
        ha='center', va='center',
        fontsize=18, fontweight='bold',
        color=text_color)

# Add subtitle
ax.text(5, 15.4, 'Pain Medication Review Analysis',
        ha='center', va='center',
        fontsize=12,
        color=label_color,
        style='italic')

# Add legend
legend_y = -2.8
legend_elements = [
    mpatches.Patch(facecolor='#D4E6F1', edgecolor='#2874A6', linewidth=2, label='Data Source'),
    mpatches.Patch(facecolor='#FEF5E7', edgecolor='#D68910', linewidth=2, label='Processing Step'),
    mpatches.Patch(facecolor='#E8F4F8', edgecolor='#2C5F7C', linewidth=2, label='Data State'),
    mpatches.Patch(facecolor='#D5F4E6', edgecolor='#229954', linewidth=2, label='Final Output')
]

ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.5, -0.15), ncol=4,
          frameon=True, fontsize=9, edgecolor='gray')

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = '/Users/macbook/CADT/Term2Year3/Data/Final_Project/project/05_analysis_results/plots/data_flow_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Data flow diagram saved to: {output_path}")
plt.close()

print("\nDiagram created successfully!")
print(f"Resolution: 300 DPI")
print(f"Format: PNG")
print(f"Dimensions: 12x14 inches")
