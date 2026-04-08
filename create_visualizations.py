import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory if it doesn't exist
output_dir = '/Users/macbook/CADT/Term2Year3/Data/Final_Project/project/05_analysis_results/plots'
os.makedirs(output_dir, exist_ok=True)

# ============================
# IMAGE 1: Random Forest Architecture
# ============================
def create_random_forest_architecture():
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Random Forest Classifier Architecture', 
            ha='center', va='top', fontsize=24, fontweight='bold')
    
    # Input features box (left)
    input_box = FancyBboxPatch((0.5, 3.5), 1.5, 3, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#2E86AB', facecolor='#A9D6E5', 
                                linewidth=3)
    ax.add_patch(input_box)
    ax.text(1.25, 6.5, 'Input Features', ha='center', va='top', 
            fontsize=14, fontweight='bold')
    ax.text(1.25, 6.1, '(67 Features)', ha='center', va='top', fontsize=11)
    
    # Feature names
    features = ['usefulCount', 'compound', 'review_length', 'pos', 'neg', '...']
    for i, feat in enumerate(features):
        ax.text(1.25, 5.5 - i*0.3, feat, ha='center', va='center', 
                fontsize=9, style='italic')
    
    # Decision Trees (middle) - 5 trees
    tree_positions = [2.5, 3.5, 4.5, 5.5, 6.5]
    tree_colors = ['#013A63', '#01497C', '#014F86', '#2C7DA0', '#468FAF']
    
    for i, (pos, color) in enumerate(zip(tree_positions, tree_colors)):
        # Tree triangle
        tree_x = [pos, pos-0.35, pos+0.35, pos]
        tree_y = [7, 4.5, 4.5, 7]
        ax.fill(tree_x, tree_y, color=color, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Tree trunk
        trunk = Rectangle((pos-0.1, 3.5), 0.2, 1, 
                          facecolor='#8B4513', edgecolor='black', linewidth=1.5)
        ax.add_patch(trunk)
        
        # Tree label
        ax.text(pos, 3.2, f'Tree {i+1}', ha='center', va='top', 
                fontsize=9, fontweight='bold')
        
        # Prediction from each tree
        predictions = ['Not Eff.', 'Effective', 'Part. Eff.', 'Effective', 'Effective']
        ax.text(pos, 2.8, predictions[i], ha='center', va='top', 
                fontsize=8, style='italic', color=color)
    
    # Arrow from input to trees
    arrow1 = FancyArrowPatch((2, 5.5), (2.3, 6), 
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=3, color='#555555')
    ax.add_patch(arrow1)
    
    # Voting mechanism box
    vote_box = FancyBboxPatch((3.5, 1.5), 3, 1.2, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='#E63946', facecolor='#F4A261', 
                               linewidth=3)
    ax.add_patch(vote_box)
    ax.text(5, 2.4, 'Voting Mechanism', ha='center', va='center', 
            fontsize=13, fontweight='bold')
    ax.text(5, 2.0, 'Majority Vote', ha='center', va='center', 
            fontsize=11, style='italic')
    ax.text(5, 1.7, '(Aggregate Predictions)', ha='center', va='center', 
            fontsize=9)
    
    # Arrows from trees to voting
    for pos in tree_positions:
        arrow = FancyArrowPatch((pos, 2.7), (pos, 2.8), 
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='#555555')
        ax.add_patch(arrow)
    
    # Final prediction box (right)
    output_box = FancyBboxPatch((7.5, 4), 2, 2.5, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='#2A9D8F', facecolor='#8ECAE6', 
                                 linewidth=3)
    ax.add_patch(output_box)
    ax.text(8.5, 6.2, 'Final Prediction', ha='center', va='top', 
            fontsize=14, fontweight='bold')
    ax.text(8.5, 5.7, 'Effective', ha='center', va='center', 
            fontsize=16, fontweight='bold', color='#155724',
            bbox=dict(boxstyle='round', facecolor='#D4EDDA', edgecolor='#155724', linewidth=2))
    ax.text(8.5, 5.0, 'Classification:', ha='center', va='center', fontsize=10)
    ax.text(8.5, 4.7, '• Not Effective', ha='left', va='center', fontsize=9)
    ax.text(8.5, 4.4, '• Partially Effective', ha='left', va='center', fontsize=9)
    ax.text(8.5, 4.1, '• Effective', ha='left', va='center', fontsize=9)
    
    # Arrow from voting to output
    arrow3 = FancyArrowPatch((6.5, 2.1), (7.5, 5), 
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=3, color='#555555')
    ax.add_patch(arrow3)
    
    # Model configuration box at bottom
    config_box = FancyBboxPatch((2, 0.2), 6, 0.8, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='#6A4C93', facecolor='#E5D4ED', 
                                linewidth=2)
    ax.add_patch(config_box)
    ax.text(5, 0.7, '100 Trees  •  Max Depth: 20  •  SMOTE Balancing', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'random_forest_architecture.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

# ============================
# IMAGE 2: Confusion Matrix
# ============================
def create_confusion_matrix():
    # Load test predictions
    df = pd.read_csv('/Users/macbook/CADT/Term2Year3/Data/Final_Project/project/04_trained_models/test_results/test_predictions_expanded.csv')
    
    # Calculate confusion matrix
    cm = confusion_matrix(df['actual'], df['predicted'])
    accuracy = accuracy_score(df['actual'], df['predicted'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Create heatmap
    class_names = ['Not Effective', 'Partially Effective', 'Effective']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax,
                linewidths=2, linecolor='white',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    # Labels and title
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Class', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix - Model Predictions', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Add accuracy annotation
    accuracy_text = f'Overall Accuracy: {accuracy*100:.2f}%'
    ax.text(1.5, -0.3, accuracy_text, ha='center', va='top', 
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#D4EDDA', 
                     edgecolor='#155724', linewidth=2))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

# ============================
# IMAGE 3: Feature Importance
# ============================
def create_feature_importance():
    # Load feature importance
    df = pd.read_csv('/Users/macbook/CADT/Term2Year3/Data/Final_Project/project/04_trained_models/random_forest_v2_expanded/feature_importance.csv')
    
    # Get top 10 features
    df_top10 = df.nlargest(10, 'importance')
    
    # Convert to percentage
    df_top10['importance_pct'] = df_top10['importance'] * 100
    
    # Sort by importance (ascending for horizontal bar chart)
    df_top10 = df_top10.sort_values('importance_pct', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_top10)))
    
    # Create horizontal bar chart
    bars = ax.barh(df_top10['feature'], df_top10['importance_pct'], 
                   color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df_top10['importance_pct'])):
        ax.text(value + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{value:.2f}%', 
                va='center', ha='left', fontsize=11, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('Importance (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Feature Name', fontsize=14, fontweight='bold')
    ax.set_title('Top 10 Feature Importance', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_importance_top10.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path

# ============================
# Main execution
# ============================
if __name__ == '__main__':
    print("Creating visualizations...")
    
    print("\n1. Creating Random Forest Architecture diagram...")
    path1 = create_random_forest_architecture()
    size1 = os.path.getsize(path1) / 1024  # KB
    print(f"   ✓ Saved to: {path1}")
    print(f"   ✓ Size: {size1:.2f} KB")
    
    print("\n2. Creating Confusion Matrix heatmap...")
    path2 = create_confusion_matrix()
    size2 = os.path.getsize(path2) / 1024  # KB
    print(f"   ✓ Saved to: {path2}")
    print(f"   ✓ Size: {size2:.2f} KB")
    
    print("\n3. Creating Feature Importance chart...")
    path3 = create_feature_importance()
    size3 = os.path.getsize(path3) / 1024  # KB
    print(f"   ✓ Saved to: {path3}")
    print(f"   ✓ Size: {size3:.2f} KB")
    
    print("\n" + "="*60)
    print("All visualizations created successfully!")
    print("="*60)
