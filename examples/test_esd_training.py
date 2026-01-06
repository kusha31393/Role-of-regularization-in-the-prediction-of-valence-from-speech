#!/usr/bin/env python3
"""
Quick test script to verify ESD dataset loading and training pipeline works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import load_esd_opensmile_data
import torch

def main():
    print("=" * 60)
    print("ESD Dataset + OpenSMILE Features Test")
    print("=" * 60)
    
    # Test data loading
    print("\n1. Testing ESD data loading...")
    try:
        features, labels = load_esd_opensmile_data()
        print(f" Data loaded successfully!")
        print(f"   Features shape: {features.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Feature type: {features.dtype}")
        
        # Check for any NaN values
        nan_features = torch.isnan(torch.from_numpy(features)).any()
        nan_labels = labels.isnull().any().any()
        
        print(f"   NaN in features: {nan_features}")
        print(f"   NaN in labels: {nan_labels}")
        
        print(f"\n   Sample label stats:")
        for col in ['valence', 'arousal', 'dominance']:
            print(f"   {col}: min={labels[col].min():.3f}, max={labels[col].max():.3f}, mean={labels[col].mean():.3f}")
            
    except Exception as e:
        print(f" Data loading failed: {e}")
        return
    
    # Test basic model creation
    print("\n2. Testing model creation...")
    try:
        from model import EmotionMLP
        
        model = EmotionMLP(
            input_size=features.shape[1],  # 6373 for OpenSMILE
            hidden_sizes=[512, 256],
            dropout_rate=0.3
        )
        
        print(f" Model created successfully!")
        print(f"   Input dim: {features.shape[1]}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with a small batch
        test_batch = torch.from_numpy(features[:10])  # First 10 samples
        with torch.no_grad():
            output = model(test_batch)
        
        print(f"   Forward pass successful: {test_batch.shape} -> {output.shape}")
        
    except Exception as e:
        print(f" Model test failed: {e}")
        return
    
    print("\n3. Testing loss function...")
    try:
        from loss import CCCLoss
        
        loss_fn = CCCLoss()
        target_batch = torch.from_numpy(labels['valence'].values[:10]).float().unsqueeze(1)
        
        loss = loss_fn(output, target_batch)
        print(f" Loss computation successful: {loss.item():.4f}")
        
    except Exception as e:
        print(f" Loss test failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print(" All tests passed! Ready for training with ESD + OpenSMILE")
    print("=" * 60)
    
    print(f"\nTo run a quick training test:")
    print(f"python src/train.py --use_esd_data --experiment_type dropout --device cpu --max_epochs 2")

if __name__ == '__main__':
    main()