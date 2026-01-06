#!/usr/bin/env python3
"""
Complete Training Example for Speech Emotion Recognition with OpenSMILE Features

This script demonstrates the complete pipeline:
1. Loading ESD dataset with OpenSMILE ComParE features
2. Training neural networks for emotion recognition
3. Generating loss curves and performance plots
4. Saving results for GitHub repository demonstration

Usage:
    python run_example_training.py

This will:
- Train for 100 epochs with validation every 5 epochs  
- Test 4 different dropout rates (0.0, 0.3, 0.5, 0.7)
- Focus on valence prediction (most challenging emotion)
- Generate comprehensive plots and results
- Save everything in experiments/ directory

Requirements:
- ESD dataset with OpenSMILE features (run extract_opensmile_features.py first)
- ~10-15 minutes on CPU (depending on hardware)
"""

import os
import sys
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from train import main as train_main

def create_example_run():
    """Execute the complete example training run"""
    
    print("=" * 80)
    print(" SPEECH EMOTION RECOGNITION - COMPLETE TRAINING EXAMPLE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set up arguments for the training script
    class Args:
        experiment = 'dropout'
        data_dir = None
        use_dummy_data = False
        use_esd_data = True
        device = 'cpu'
    
    args = Args()
    
    print(f"\n Configuration:")
    print(f"   Experiment type: {args.experiment}")
    print(f"   Device: {args.device}")
    print(f"   Dataset: ESD with OpenSMILE ComParE features")
    print(f"   Training: 20 epochs, validation every 2 epochs (quick demo)")
    print(f"   Dropout rate: 0.5 (focused example)")
    print(f"   Target emotion: Valence")
    print(f"   Dataset: Full ESD dataset (2,500 samples)")
    
    print(f"\n Output will be saved to:")
    print(f"   experiments/valence_dropout_0.5/")
    print(f"   - Training curves (PNG)")
    print(f"   - Model checkpoints") 
    print(f"   - Results summaries")
    
    # Confirmation
    print(f"\n  Estimated time: 3-5 minutes on CPU")
    print(f" Disk space needed: ~10MB for results")
    
    print(f"\n Starting training automatically...")
    
    print(f"\n" + "=" * 80)
    print(" STARTING TRAINING...")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Temporarily replace sys.argv to pass arguments
        original_argv = sys.argv
        sys.argv = ['train.py', 
                   '--experiment', args.experiment, 
                   '--use_esd_data',
                   '--device', args.device,
                   '--dropout_rate', '0.5',
                   '--epochs', '20',
                   '--validation_frequency', '2']
        
        # Run the training
        train_main()
        
    except KeyboardInterrupt:
        print(f"\n  Training interrupted by user")
        return
    except Exception as e:
        print(f"\n Training failed: {e}")
        return
    finally:
        # Restore sys.argv
        sys.argv = original_argv
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n" + "=" * 80)
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"  Total time: {duration/60:.1f} minutes")
    print(f" Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check what was created
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        print(f"\n Generated files:")
        
        # Find dropout experiment directories
        dropout_dirs = list(experiments_dir.glob("valence_dropout_*"))
        
        total_checkpoints = 0
        total_plots = 0
        
        if dropout_dirs:
            for dropout_dir in sorted(dropout_dirs):
                dropout_rate = dropout_dir.name.split('_')[-1]
                print(f"\n    {dropout_dir.name}/")
                
                # List files in each directory
                for file_path in sorted(dropout_dir.glob("*")):
                    size_mb = file_path.stat().st_size / (1024*1024)
                    
                    # Count different file types
                    if file_path.suffix == '.pth':
                        total_checkpoints += 1
                        print(f"       {file_path.name} ({size_mb:.1f}MB) - Model checkpoint")
                    elif file_path.suffix == '.png':
                        total_plots += 1
                        print(f"       {file_path.name} ({size_mb:.1f}MB) - Training curves")
                    else:
                        print(f"       {file_path.name} ({size_mb:.1f}MB)")
        
        print(f"\n Summary:")
        print(f"    {total_plots} training curve plots generated")
        print(f"    {total_checkpoints} model checkpoints saved")
        print(f"    Results data saved for analysis")
        
        # Show how to load checkpoints
        if total_checkpoints > 0:
            print(f"\n To load a checkpoint for analysis:")
            print(f"   from src.train import load_checkpoint, list_available_checkpoints")
            print(f"   from src.model import EmotionMLP")
            print(f"   ")
            print(f"   # List all available checkpoints")
            print(f"   checkpoints = list_available_checkpoints()")
            print(f"   ")
            print(f"   # Load best model for dropout 0.5")
            print(f"   model = EmotionMLP(input_size=6373, hidden_sizes=[256, 256], dropout_rate=0.5)")
            print(f"   checkpoint = load_checkpoint('experiments/valence_dropout_0.5/best_model_dropout_0.5.pth', model)")
            print(f"   ")
            print(f"   # Access training history")
            print(f"   train_history = checkpoint['train_ccc_history']")
            print(f"   val_history = checkpoint['val_ccc_history']")
    
    print(f"\n Example complete! You can now:")
    print(f"   1. View the training curve plots in experiments/")
    print(f"   2. Analyze the results data")
    print(f"   3. Use this as a reference for your own experiments")
    print(f"   4. Push to GitHub as a working example")
    
    print(f"\n GitHub repository ready with:")
    print(f"    Complete working code")
    print(f"    Example configuration") 
    print(f"    Sample results and plots")
    print(f"    Documentation and usage instructions")


if __name__ == '__main__':
    # Change to script directory to ensure relative paths work
    os.chdir(Path(__file__).parent)
    
    try:
        create_example_run()
    except KeyboardInterrupt:
        print(f"\n👋 Goodbye!")
    except Exception as e:
        print(f"\n Error: {e}")
        print(f" Make sure you've run extract_opensmile_features.py first!")