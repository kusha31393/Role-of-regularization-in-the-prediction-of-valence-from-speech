#!/usr/bin/env python3
"""
Test script for loss curve plotting functionality
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils import plot_training_curves, plot_loss_curves

def create_mock_training_data():
    """Create mock training data that looks realistic"""
    epochs = 20
    
    # Simulate decreasing loss with some noise
    train_losses = []
    val_losses = []
    train_cccs = []
    val_cccs = []
    
    for epoch in range(epochs):
        # Training loss decreases with noise
        train_loss = 0.8 - 0.3 * (epoch / epochs) + 0.05 * np.random.randn()
        train_ccc = 1 - train_loss  # CCC = 1 - loss
        
        # Validation loss decreases slower with more noise  
        val_loss = 0.85 - 0.25 * (epoch / epochs) + 0.08 * np.random.randn()
        val_ccc = 1 - val_loss
        
        # Add some overfitting after epoch 15
        if epoch > 15:
            val_loss += 0.02 * (epoch - 15)
            val_ccc = 1 - val_loss
        
        train_losses.append(max(0.1, train_loss))  # Prevent negative loss
        val_losses.append(max(0.1, val_loss))
        train_cccs.append(min(0.9, max(0.1, train_ccc)))  # Clamp CCC
        val_cccs.append(min(0.9, max(0.1, val_ccc)))
    
    return train_losses, val_losses, train_cccs, val_cccs

def main():
    print("=" * 60)
    print("Testing Loss Curve Plotting Functionality")
    print("=" * 60)
    
    # Create mock training data
    train_losses, val_losses, train_cccs, val_cccs = create_mock_training_data()
    
    print(f"Created mock training data:")
    print(f"  Epochs: {len(train_losses)}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}")
    print(f"  Final train CCC: {train_cccs[-1]:.4f}")
    print(f"  Final val CCC: {val_cccs[-1]:.4f}")
    
    # Test 1: Plot individual training curves
    print("\n1. Testing individual curve plotting...")
    try:
        plot_training_curves(
            train_cccs, val_cccs, 
            metric_name='CCC',
            title='Test: Training and Validation CCC',
            early_stop_epoch=16,
            save_path='test_ccc_curves.png'
        )
        print(" Individual CCC curve plotting successful!")
    except Exception as e:
        print(f" Individual curve plotting failed: {e}")
        return
    
    # Test 2: Plot combined loss curves  
    print("\n2. Testing combined loss curve plotting...")
    try:
        plot_loss_curves(
            train_losses, val_losses,
            train_cccs, val_cccs,
            early_stop_epoch=16,
            save_path='test_combined_curves.png'
        )
        print(" Combined loss curve plotting successful!")
    except Exception as e:
        print(f" Combined curve plotting failed: {e}")
        return
    
    print(f"\n" + "=" * 60)
    print(" All loss curve plotting tests passed!")
    print(" Check generated files: test_ccc_curves.png, test_combined_curves.png")
    print("=" * 60)

if __name__ == '__main__':
    main()