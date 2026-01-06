"""
Model architecture visualization script for SER regularization paper.

This script generates visualizations of the neural network architecture described in:
"Role of Regularization in the Prediction of Valence from Speech" by Sridhar et al. (2018)

Usage:
    python visualize_model.py --arch          # Main architecture diagram only
    python visualize_model.py --variations    # Architecture variations only
    python visualize_model.py --dropout       # Dropout analysis chart only
    python visualize_model.py --all           # All visualizations
    python visualize_model.py                 # Default: all visualizations

Generated Files:
    - experiments/model_architecture.png: Main DNN architecture diagram
    - experiments/architecture_variations.png: Different layer configurations
    - experiments/dropout_comparison.png: Dropout rate analysis chart

Architecture Details:
    - Input: 6,373 OpenSmile ComParE 2013 features
    - Hidden: 2 layers × 256 neurons each
    - Processing: Linear → BatchNorm → ReLU → Dropout
    - Output: Single continuous emotion score
    - Key finding: Valence needs higher dropout (0.7-0.8) vs arousal/dominance (0.4-0.5)
"""

import sys
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from model import EmotionMLP


def visualize_mlp_architecture(
    input_size: int = 6373,
    hidden_sizes: list = [256, 256],
    dropout_rate: float = 0.5,
    save_path: str = None
):
    """
    Visualize the MLP architecture used in the paper.
    
    Args:
        input_size: Input feature dimension
        hidden_sizes: List of hidden layer sizes
        dropout_rate: Dropout probability
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Layer information
    layers = ['Input'] + [f'Hidden {i+1}' for i in range(len(hidden_sizes))] + ['Output']
    layer_sizes = [input_size] + hidden_sizes + [1]
    
    # Position parameters
    layer_width = 1.0
    layer_spacing = 0.8
    max_neurons_display = 4  # Maximum neurons to show visually
    
    # Colors
    colors = {
        'input': '#E8F4FD',
        'hidden': '#B3D9FF',
        'output': '#FF9999',
        'dropout': '#FFE6CC',
        'batch_norm': '#E6FFE6'
    }
    
    # Draw layers
    for i, (layer_name, size) in enumerate(zip(layers, layer_sizes)):
        x_pos = i * layer_spacing
        
        # Determine how many neurons to show
        neurons_to_show = min(size, max_neurons_display)
        neuron_spacing = 0.25
        total_height = neurons_to_show * neuron_spacing
        y_start = -total_height / 2
        
        # Layer color
        if i == 0:
            color = colors['input']
        elif i == len(layers) - 1:
            color = colors['output']
        else:
            color = colors['hidden']
        
        # Draw neurons as smaller circles
        for j in range(neurons_to_show):
            y_pos = y_start + j * neuron_spacing
            
            # Draw neuron circle
            circle = plt.Circle((x_pos, y_pos), 0.08, facecolor=color, 
                              edgecolor='black', linewidth=0.8)
            ax.add_patch(circle)
        
        # No "..." indicators - just show the representative neurons
        
        # Layer label with size - closer to neurons
        if size > 1000:
            size_text = f'{size//1000}k'
        else:
            size_text = str(size)
        
        layer_label = f'{layer_name} ({size_text})'
        ax.text(x_pos, -total_height/2 - 0.3, layer_label, ha='center', 
               va='center', fontsize=9, weight='normal')
        
        # Add processing blocks for hidden layers - closer to neurons
        if 0 < i < len(layers) - 1:  # Hidden layers only
            # Batch normalization - closer to neurons
            bn_box = FancyBboxPatch(
                (x_pos - 0.3, total_height/2 + 0.15), 0.6, 0.15,
                boxstyle="round,pad=0.02", 
                facecolor=colors['batch_norm'], 
                edgecolor='green', 
                linewidth=1
            )
            ax.add_patch(bn_box)
            ax.text(x_pos, total_height/2 + 0.225, 'BatchNorm', ha='center', 
                   va='center', fontsize=6)
            
            # ReLU activation
            relu_box = FancyBboxPatch(
                (x_pos - 0.25, total_height/2 + 0.32), 0.5, 0.12,
                boxstyle="round,pad=0.02",
                facecolor='lightblue',
                edgecolor='blue',
                linewidth=1
            )
            ax.add_patch(relu_box)
            ax.text(x_pos, total_height/2 + 0.38, 'ReLU', ha='center', 
                   va='center', fontsize=6, weight='bold')
            
            # Dropout - closer to other boxes
            dropout_box = FancyBboxPatch(
                (x_pos - 0.35, total_height/2 + 0.46), 0.7, 0.16,
                boxstyle="round,pad=0.02",
                facecolor=colors['dropout'],
                edgecolor='orange',
                linewidth=1.5
            )
            ax.add_patch(dropout_box)
            ax.text(x_pos, total_height/2 + 0.54, f'Dropout({dropout_rate})', 
                   ha='center', va='center', fontsize=6, weight='bold')
    
    # Draw fully connected lines between displayed neurons
    for i in range(len(layers) - 1):
        x_start = i * layer_spacing
        x_end = (i + 1) * layer_spacing
        
        # Get positions of neurons in current and next layer
        neurons_current = min(layer_sizes[i], max_neurons_display)
        neurons_next = min(layer_sizes[i+1], max_neurons_display)
        
        current_spacing = 0.25
        next_spacing = 0.25
        current_height = neurons_current * current_spacing
        next_height = neurons_next * next_spacing
        
        current_y_start = -current_height / 2
        next_y_start = -next_height / 2
        
        # Draw line from each neuron to every neuron in next layer
        for j in range(neurons_current):
            y_current = current_y_start + j * current_spacing
            for k in range(neurons_next):
                y_next = next_y_start + k * next_spacing
                ax.plot([x_start, x_end], [y_current, y_next], 'k-', alpha=0.2, linewidth=0.3)
    
    # Add title and labels (more compact)
    ax.set_title(f'DNN Architecture for Speech Emotion Recognition', 
                fontsize=12, weight='bold', pad=15)
    
    # Add compact legend on the right side
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input (6k features)'),
        patches.Patch(color=colors['hidden'], label='Hidden (256 neurons)'),
        patches.Patch(color=colors['output'], label='Output (1 score)'),
        patches.Patch(color=colors['dropout'], label=f'Dropout (p={dropout_rate})'),
    ]
    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.05, 0.95), fontsize=8)
    
    # Set axis properties - more compact width
    ax.set_xlim(-0.5, len(layers) * layer_spacing + 0.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add architecture details text - positioned to the right to avoid overlap
    details_text = f"""Key Details:
• Features: {input_size} OpenSmile ComParE 2013
• Architecture: {len(hidden_sizes)} × {hidden_sizes[0]} hidden layers
• Processing: Linear → BatchNorm → ReLU → Dropout
• Loss: CCC, Optimizer: SGD (momentum=0.9, lr=0.001)"""
    
    ax.text(len(layers) * layer_spacing - 0.5, -0.95, details_text, fontsize=8, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.9),
           ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to {save_path}")
    else:
        plt.savefig('experiments/model_architecture.png', dpi=300, bbox_inches='tight')
        print("Architecture diagram saved to experiments/model_architecture.png")
    
    plt.close()


def compare_architectures():
    """
    Compare different architectures tested in the paper.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    architectures = [
        {'layers': 2, 'nodes': 256, 'title': '2 Layers × 256 Nodes'},
        {'layers': 4, 'nodes': 256, 'title': '4 Layers × 256 Nodes'},
        {'layers': 6, 'nodes': 256, 'title': '6 Layers × 256 Nodes'}
    ]
    
    for idx, (ax, arch) in enumerate(zip(axes, architectures)):
        # Simple block representation
        layer_height = 0.8
        layer_width = 1.2
        spacing = 0.2
        
        # Input layer
        input_rect = FancyBboxPatch(
            (0, 0), layer_width, layer_height,
            boxstyle="round,pad=0.1",
            facecolor='lightblue',
            edgecolor='blue'
        )
        ax.add_patch(input_rect)
        ax.text(layer_width/2, layer_height/2, 'Input\\n(6373)', 
               ha='center', va='center', fontsize=10, weight='bold')
        
        # Hidden layers
        for i in range(arch['layers']):
            y_pos = (layer_height + spacing) * (i + 1)
            hidden_rect = FancyBboxPatch(
                (0, y_pos), layer_width, layer_height,
                boxstyle="round,pad=0.1",
                facecolor='lightgreen',
                edgecolor='green'
            )
            ax.add_patch(hidden_rect)
            ax.text(layer_width/2, y_pos + layer_height/2, 
                   f'Hidden {i+1}\\n({arch["nodes"]})', 
                   ha='center', va='center', fontsize=10, weight='bold')
        
        # Output layer
        y_pos = (layer_height + spacing) * (arch['layers'] + 1)
        output_rect = FancyBboxPatch(
            (0, y_pos), layer_width, layer_height,
            boxstyle="round,pad=0.1",
            facecolor='lightcoral',
            edgecolor='red'
        )
        ax.add_patch(output_rect)
        ax.text(layer_width/2, y_pos + layer_height/2, 'Output\\n(1)', 
               ha='center', va='center', fontsize=10, weight='bold')
        
        # Draw arrows between layers
        for i in range(arch['layers'] + 1):
            y_start = (layer_height + spacing) * i + layer_height
            y_end = (layer_height + spacing) * (i + 1)
            ax.annotate('', xy=(layer_width/2, y_end), 
                       xytext=(layer_width/2, y_start),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        ax.set_xlim(-0.2, layer_width + 0.2)
        ax.set_ylim(-0.2, y_pos + layer_height + 0.2)
        ax.set_title(arch['title'], fontsize=12, weight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.suptitle('Architecture Variations Tested in Paper', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('experiments/architecture_variations.png', dpi=300, bbox_inches='tight')
    print("Architecture variations saved to experiments/architecture_variations.png")
    plt.close()


def create_dropout_comparison():
    """
    Visualize the effect of different dropout rates.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Simulate dropout rates and performance
    dropout_rates = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Simulated CCC scores based on paper results
    valence_ccc = np.array([0.35, 0.37, 0.39, 0.41, 0.42, 0.43, 0.44, 0.448, 0.445, 0.42])
    arousal_ccc = np.array([0.65, 0.68, 0.70, 0.715, 0.717, 0.717, 0.710, 0.70, 0.68, 0.65])
    dominance_ccc = np.array([0.60, 0.64, 0.67, 0.685, 0.690, 0.692, 0.685, 0.675, 0.65, 0.62])
    
    # Plot lines
    ax.plot(dropout_rates, valence_ccc, 'r-o', linewidth=3, markersize=8, 
           label='Valence (optimal: 0.7-0.8)')
    ax.plot(dropout_rates, arousal_ccc, 'b-s', linewidth=3, markersize=8, 
           label='Arousal (optimal: 0.4-0.5)')
    ax.plot(dropout_rates, dominance_ccc, 'g-^', linewidth=3, markersize=8, 
           label='Dominance (optimal: 0.4-0.5)')
    
    # Highlight optimal regions
    ax.axvspan(0.7, 0.8, alpha=0.2, color='red', label='Valence optimal range')
    ax.axvspan(0.4, 0.5, alpha=0.2, color='blue', label='Arousal/Dominance optimal range')
    
    # Formatting
    ax.set_xlabel('Dropout Rate', fontsize=14, weight='bold')
    ax.set_ylabel('CCC (Concordance Correlation Coefficient)', fontsize=14, weight='bold')
    ax.set_title('Dropout Rate vs Performance by Emotion Attribute\\n(Expected Pattern from Paper)', 
                fontsize=16, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 0.75)
    
    # Add annotations
    ax.annotate('Valence needs higher\\nregularization', 
               xy=(0.75, 0.448), xytext=(0.6, 0.52),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'),
               fontsize=12, weight='bold', color='red',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.annotate('Arousal/Dominance\\noptimal at lower rates', 
               xy=(0.45, 0.715), xytext=(0.25, 0.65),
               arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
               fontsize=12, weight='bold', color='blue',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('experiments/dropout_comparison.png', dpi=300, bbox_inches='tight')
    print("Dropout comparison saved to experiments/dropout_comparison.png")
    plt.close()


def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Generate neural network architecture visualizations for the SER regularization paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_model.py --arch              # Generate only the main architecture diagram
  python visualize_model.py --dropout           # Generate only the dropout comparison
  python visualize_model.py --variations        # Generate only the architecture variations
  python visualize_model.py --all               # Generate all visualizations (default)
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--arch', action='store_true', 
                      help='Generate only the main architecture diagram (experiments/model_architecture.png)')
    group.add_argument('--dropout', action='store_true',
                      help='Generate only the dropout rate comparison (experiments/dropout_comparison.png)')
    group.add_argument('--variations', action='store_true',
                      help='Generate only the architecture variations (experiments/architecture_variations.png)')
    group.add_argument('--all', action='store_true',
                      help='Generate all visualizations (default behavior)')
    
    args = parser.parse_args()
    
    # Default to --all if no specific option is chosen
    if not any([args.arch, args.dropout, args.variations, args.all]):
        args.all = True
    
    # Generate requested visualizations
    if args.arch or args.all:
        print("Generating main architecture diagram...")
        visualize_mlp_architecture(save_path="experiments/model_architecture.png")
    
    if args.dropout or args.all:
        print("Generating dropout rate comparison...")
        create_dropout_comparison()
    
    if args.variations or args.all:
        print("Generating architecture variations...")
        compare_architectures()
    
    print("Visualization(s) generated successfully!")



if __name__ == '__main__':
    main()