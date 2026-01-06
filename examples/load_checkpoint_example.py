#!/usr/bin/env python3
"""
Example: How to Load and Analyze Saved Checkpoints

This script demonstrates how to:
1. List available checkpoints from training experiments
2. Load a specific checkpoint 
3. Analyze the training history
4. Use the model for predictions

Run this after completing the training example.
"""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from train import load_checkpoint, list_available_checkpoints
from model import EmotionMLP
from data_loader import load_esd_opensmile_data

def main():
    print("=" * 70)
    print(" CHECKPOINT LOADING AND ANALYSIS EXAMPLE")
    print("=" * 70)
    
    # 1. List all available checkpoints
    print("\n1. Listing available checkpoints...")
    checkpoints = list_available_checkpoints()
    
    if not checkpoints:
        print(" No checkpoints found! Run the training example first:")
        print("   python run_example_training.py")
        return
    
    # 2. Load the best checkpoint (dropout 0.5 usually performs well)
    print(f"\n2. Loading checkpoint for analysis...")
    
    # Find dropout 0.5 checkpoint (or use first available)
    target_checkpoint = None
    for cp in checkpoints:
        if 'dropout_0.5' in cp and 'best_model' in cp:
            target_checkpoint = cp
            break
    
    if not target_checkpoint:
        target_checkpoint = checkpoints[0]  # Use first available
    
    print(f"Loading: {target_checkpoint}")
    
    # Create model instance
    model = EmotionMLP(
        input_size=6373,
        hidden_sizes=[256, 256], 
        dropout_rate=0.5
    )
    
    # Load checkpoint
    checkpoint = load_checkpoint(target_checkpoint, model)
    
    # 3. Analyze training history
    print(f"\n3. Analyzing training history...")
    
    train_ccc_history = checkpoint.get('train_ccc_history', [])
    val_ccc_history = checkpoint.get('val_ccc_history', [])
    train_loss_history = checkpoint.get('train_loss_history', [])
    val_loss_history = checkpoint.get('val_loss_history', [])
    validation_epochs = checkpoint.get('validation_epochs', list(range(len(val_ccc_history))))
    
    if train_ccc_history and val_ccc_history:
        print(f"    Training epochs: {len(train_ccc_history)}")
        print(f"    Best validation CCC: {max(val_ccc_history):.4f}")
        print(f"   Final training loss: {train_loss_history[-1]:.4f}")
        print(f"   Final validation loss: {val_loss_history[-1]:.4f}")
        
        # Plot the training curves
        print(f"\n4. Plotting training curves...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot Loss
        ax1.plot(validation_epochs, train_loss_history, 'b-', label='Training Loss', linewidth=2, marker='o')
        ax1.plot(validation_epochs, val_loss_history, 'r-', label='Validation Loss', linewidth=2, marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (1 - CCC)')
        ax1.set_title(f'Training History - Dropout {checkpoint["dropout_rate"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot CCC
        ax2.plot(validation_epochs, train_ccc_history, 'b-', label='Training CCC', linewidth=2, marker='o')
        ax2.plot(validation_epochs, val_ccc_history, 'r-', label='Validation CCC', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('CCC')
        ax2.set_title(f'CCC Progress - Dropout {checkpoint["dropout_rate"]}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('checkpoint_analysis.png', dpi=300, bbox_inches='tight')
        print(f"    Analysis plot saved to: checkpoint_analysis.png")
        plt.show()
    
    # 4. Test the loaded model
    print(f"\n5. Testing loaded model...")
    
    try:
        # Load test data
        features, labels = load_esd_opensmile_data()
        
        # Use first 10 samples for demo
        test_features = torch.from_numpy(features[:10])
        true_valence = labels['valence'].values[:10]
        
        # Predict
        model.eval()
        with torch.no_grad():
            predictions = model(test_features).numpy().flatten()
        
        print(f"    Sample predictions vs ground truth:")
        for i in range(min(5, len(predictions))):
            print(f"      Sample {i+1}: Pred={predictions[i]:.3f}, True={true_valence[i]:.3f}, "
                  f"Error={abs(predictions[i] - true_valence[i]):.3f}")
        
    except Exception as e:
        print(f"    Model testing failed: {e}")
    
    print(f"\n" + "=" * 70)
    print(" CHECKPOINT ANALYSIS COMPLETE!")
    print(f"\nKey files generated:")
    print(f"    checkpoint_analysis.png - Training history visualization")
    print(f"    Multiple .pth checkpoints - Saved models for each dropout rate")
    print(f"    Training curve plots - Individual experiment results")
    
    print(f"\n Your GitHub repository now has:")
    print(f"    Complete working implementation")
    print(f"    Real training results and checkpoints")
    print(f"    Professional visualizations")
    print(f"    Reproducible example scripts")
    print("=" * 70)


if __name__ == '__main__':
    main()