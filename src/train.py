"""
Training script for speech emotion recognition experiments.
Implements the experimental framework from the paper.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model import EmotionMLP, create_model_from_config
from loss import CCCLoss, get_loss_function, evaluate_metrics
from data_loader import (
    load_msp_podcast_data, create_data_loaders, create_speaker_splits,
    create_speaker_dependent_split, create_dummy_data, get_paper_data_splits,
    load_esd_opensmile_data
)
from utils import (
    set_seed, load_config, create_experiment_dir, ExperimentLogger,
    plot_dropout_analysis, analyze_optimal_dropout_rates, save_experiment_summary,
    statistical_significance_test, plot_training_curves, plot_loss_curves
)


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: str = 'cpu'):
    """
    Load a saved checkpoint and restore model state
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model instance to load weights into
        device: Device to load the checkpoint on
    
    Returns:
        Dictionary with checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best Val CCC: {checkpoint['best_val_ccc']:.4f}")
    print(f"   Attribute: {checkpoint['attribute']}")
    print(f"   Dropout Rate: {checkpoint['dropout_rate']}")
    
    return checkpoint


def list_available_checkpoints(experiments_dir: str = "experiments"):
    """
    List all available checkpoints in the experiments directory
    
    Args:
        experiments_dir: Directory containing experiment results
    
    Returns:
        List of checkpoint paths
    """
    exp_path = Path(experiments_dir)
    if not exp_path.exists():
        print(f"No experiments directory found at: {exp_path}")
        return []
    
    checkpoints = []
    for checkpoint_file in exp_path.rglob("*.pth"):
        checkpoints.append(str(checkpoint_file))
    
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoints:")
        for cp in sorted(checkpoints):
            print(f"  {cp}")
    else:
        print("No checkpoints found")
    
    return sorted(checkpoints)


class Trainer:
    """
    Trainer class for emotion recognition experiments.
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            device: Training device ('cuda' or 'cpu')
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        print(f"Using device: {self.device}")
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            epoch: Current epoch number
        
        Returns:
            Dictionary with training metrics
        """
        model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch_idx, (features, targets) in enumerate(pbar):
            features, targets = features.to(self.device), targets.to(self.device)
            targets = targets.squeeze(-1)  # Remove extra dimension
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(features)
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            all_predictions.append(predictions.detach())
            all_targets.append(targets.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        metrics = evaluate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def evaluate(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        Evaluate model on validation or test set.
        
        Args:
            model: Neural network model
            data_loader: Data loader
            criterion: Loss function
        
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in data_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                targets = targets.squeeze(-1)
                
                # Forward pass
                predictions = model(features)
                
                # Calculate loss
                loss = criterion(predictions, targets)
                
                # Accumulate metrics
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        metrics = evaluate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(data_loader)
        
        return metrics, all_predictions.cpu(), all_targets.cpu()
    
    def train_model(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        attribute: str,
        dropout_rate: float,
        experiment_config: Dict
    ) -> Dict[str, float]:
        """
        Train a single model with given configuration.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader 
            test_loader: Test data loader
            attribute: Emotion attribute being predicted
            dropout_rate: Dropout rate used
            experiment_config: Experiment configuration
        
        Returns:
            Dictionary with final results
        """
        # Setup training
        criterion = get_loss_function(self.config['training']['loss_function'])
        
        optimizer_type = self.config['training']['optimizer'].lower()
        if optimizer_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=self.config['training']['momentum']
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['training']['learning_rate']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Training parameters
        epochs = self.config['training']['epochs']
        early_stopping = self.config['training'].get('early_stopping', {})
        patience = early_stopping.get('patience', 100)
        val_freq = self.config['training'].get('validation_frequency', 1)  # Validate every N epochs
        
        # Training loop
        best_val_ccc = -float('inf')
        best_epoch = 0
        patience_counter = 0
        
        train_ccc_history = []
        val_ccc_history = []
        train_loss_history = []
        val_loss_history = []
        validation_epochs = []  # Track which epochs had validation
        
        print(f"Training for {epochs} epochs with validation every {val_freq} epochs")
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validate only at specified frequency
            if epoch % val_freq == 0 or epoch == epochs - 1:
                val_metrics, _, _ = self.evaluate(model, val_loader, criterion)
                
                # Track history (both CCC and loss)
                train_ccc_history.append(train_metrics['ccc'])
                val_ccc_history.append(val_metrics['ccc'])
                train_loss_history.append(train_metrics['loss'])
                val_loss_history.append(val_metrics['loss'])
                validation_epochs.append(epoch)
                
                print(f"Epoch {epoch:3d}: Train CCC={train_metrics['ccc']:.4f}, Train Loss={train_metrics['loss']:.4f}, "
                      f"Val CCC={val_metrics['ccc']:.4f}, Val Loss={val_metrics['loss']:.4f}")
                # Early stopping check (only when we have validation results)
                if val_metrics['ccc'] > best_val_ccc:
                    best_val_ccc = val_metrics['ccc']
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                    
                    # Save checkpoint to disk
                    checkpoint_dir = Path(f"experiments/{attribute}_dropout_{dropout_rate}")
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_ccc': best_val_ccc,
                        'train_ccc_history': train_ccc_history,
                        'val_ccc_history': val_ccc_history,
                        'train_loss_history': train_loss_history,
                        'val_loss_history': val_loss_history,
                        'config': self.config,
                        'attribute': attribute,
                        'dropout_rate': dropout_rate
                    }
                    
                    checkpoint_path = checkpoint_dir / f"best_model_dropout_{dropout_rate}.pth"
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint: {checkpoint_path} (CCC: {best_val_ccc:.4f})")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                # Just print training progress without validation
                if epoch % 10 == 0:  # Print every 10 epochs
                    print(f"Epoch {epoch:3d}: Train CCC={train_metrics['ccc']:.4f}, Train Loss={train_metrics['loss']:.4f}")
        
        # Load best model for final evaluation
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        val_metrics, val_pred, val_true = self.evaluate(model, val_loader, criterion)
        test_metrics, test_pred, test_true = self.evaluate(model, test_loader, criterion)
        
        # Plot training curves
        if train_loss_history and val_loss_history:
            # Create experiment directory for saving plots
            exp_dir = Path(f"experiments/{attribute}_dropout_{dropout_rate}")
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create custom plot for validation frequency case
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot Loss (1 - CCC)
            ax1.plot(validation_epochs, train_loss_history, 'b-', label='Training Loss', linewidth=2, marker='o')
            ax1.plot(validation_epochs, val_loss_history, 'r-', label='Validation Loss', linewidth=2, marker='s')
            
            if best_epoch is not None:
                ax1.axvline(x=best_epoch, color='orange', linestyle='--', 
                           label=f'Best Model (Epoch {best_epoch})', alpha=0.7)
            
            # Mark best validation loss
            if val_loss_history:
                best_val_loss_idx = np.argmin(val_loss_history)
                best_val_loss = min(val_loss_history)
                ax1.plot(validation_epochs[best_val_loss_idx], best_val_loss, 'ro', markersize=8,
                        label=f'Best Val Loss: {best_val_loss:.4f}')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss (1 - CCC)')
            ax1.set_title(f'Training and Validation Loss - {attribute.title()} (Dropout {dropout_rate})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot CCC
            ax2.plot(validation_epochs, train_ccc_history, 'b-', label='Training CCC', linewidth=2, marker='o')
            ax2.plot(validation_epochs, val_ccc_history, 'r-', label='Validation CCC', linewidth=2, marker='s')
            
            if best_epoch is not None:
                ax2.axvline(x=best_epoch, color='orange', linestyle='--', 
                           label=f'Best Model (Epoch {best_epoch})', alpha=0.7)
            
            # Mark best validation CCC
            if val_ccc_history:
                best_val_ccc_idx = np.argmax(val_ccc_history)
                best_val_ccc_value = max(val_ccc_history)
                ax2.plot(validation_epochs[best_val_ccc_idx], best_val_ccc_value, 'ro', markersize=8,
                        label=f'Best Val CCC: {best_val_ccc_value:.4f}')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('CCC')
            ax2.set_title(f'Training and Validation CCC - {attribute.title()} (Dropout {dropout_rate})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = str(exp_dir / f"{attribute}_dropout_{dropout_rate}_curves.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to: {plot_path}")
            plt.close()
        
        # Save final results and model
        if train_loss_history and val_loss_history:
            results_dir = Path(f"experiments/{attribute}_dropout_{dropout_rate}")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save final model state (best model is already saved above)
            final_checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'best_val_ccc': best_val_ccc,
                'final_val_ccc': val_metrics['ccc'],
                'final_test_ccc': test_metrics['ccc'],
                'train_ccc_history': train_ccc_history,
                'val_ccc_history': val_ccc_history,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'validation_epochs': validation_epochs,
                'config': self.config,
                'attribute': attribute,
                'dropout_rate': dropout_rate
            }
            
            final_path = results_dir / f"final_results_dropout_{dropout_rate}.pth"
            torch.save(final_checkpoint, final_path)
            print(f"Saved final results: {final_path}")
        
        # Results summary
        results = {
            'attribute': attribute,
            'dropout_rate': dropout_rate,
            'best_epoch': best_epoch,
            'ccc_val': val_metrics['ccc'],
            'ccc_test': test_metrics['ccc'],
            'pearson_val': val_metrics['pearson'],
            'pearson_test': test_metrics['pearson'],
            'mse_val': val_metrics['mse'],
            'mse_test': test_metrics['mse'],
            'train_ccc_history': train_ccc_history,
            'val_ccc_history': val_ccc_history,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            **experiment_config
        }
        
        return results
    
    def run_dropout_experiment(
        self,
        features: np.ndarray,
        labels: pd.DataFrame,
        attribute: str,
        speaker_dependent: bool = False
    ) -> List[Dict[str, float]]:
        """
        Run dropout rate experiment for one emotion attribute.
        
        Args:
            features: Feature matrix
            labels: Labels DataFrame
            attribute: Emotion attribute ('valence', 'arousal', 'dominance')
            speaker_dependent: Whether to use speaker-dependent split
        
        Returns:
            List of experiment results
        """
        print(f"\nRunning dropout experiment for {attribute}")
        print(f"Speaker dependent: {speaker_dependent}")
        
        # Get speaker splits
        train_speakers, val_speakers, test_speakers = get_paper_data_splits(labels)
        
        if speaker_dependent:
            train_indices, val_indices, test_indices = create_speaker_dependent_split(
                labels, test_speakers, val_speakers, fraction=0.5
            )
        else:
            train_indices, val_indices, test_indices = create_speaker_splits(
                labels, train_speakers, val_speakers, test_speakers
            )
        
        print(f"Train samples: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # Experiment parameters
        dropout_rates = self.config['experiments']['dropout_rates']
        num_runs = self.config['experiments']['num_runs']
        
        all_results = []
        
        for dropout_rate in dropout_rates:
            print(f"\\nTesting dropout rate: {dropout_rate}")
            
            run_results = []
            
            for run in range(num_runs):
                print(f"Run {run + 1}/{num_runs}", end=" ")
                
                # Create data loaders
                train_loader, val_loader, test_loader, scaler = create_data_loaders(
                    features, labels, train_indices, val_indices, test_indices,
                    attribute=attribute,
                    batch_size=self.config['training']['batch_size'],
                    standardize=True
                )
                
                # Create model with specific dropout rate
                model_config = self.config['model'].copy()
                model_config['dropout_rate'] = dropout_rate
                
                model = EmotionMLP(
                    input_size=self.config['dataset']['feature_dim'],
                    hidden_sizes=[model_config['hidden_size']] * model_config['hidden_layers'],
                    dropout_rate=dropout_rate,
                    use_batch_norm=model_config['batch_norm']
                )
                model.to(self.device)
                
                # Train model
                experiment_config = {
                    'run': run,
                    'num_layers': model_config['hidden_layers'],
                    'hidden_size': model_config['hidden_size'],
                    'speaker_dependent': speaker_dependent
                }
                
                results = self.train_model(
                    model, train_loader, val_loader, test_loader,
                    attribute, dropout_rate, experiment_config
                )
                
                run_results.append(results)
                all_results.append(results)
                
                print(f"CCC: {results['ccc_val']:.3f} (val), {results['ccc_test']:.3f} (test)")
            
            # Statistical analysis for this dropout rate
            val_cccs = [r['ccc_val'] for r in run_results]
            test_cccs = [r['ccc_test'] for r in run_results]
            
            print(f"Dropout {dropout_rate}: Val CCC = {np.mean(val_cccs):.3f}±{np.std(val_cccs):.3f}, "
                  f"Test CCC = {np.mean(test_cccs):.3f}±{np.std(test_cccs):.3f}")
        
        return all_results
    
    def run_architecture_experiment(
        self,
        features: np.ndarray,
        labels: pd.DataFrame
    ) -> List[Dict[str, float]]:
        """
        Run experiments with different network architectures.
        
        Args:
            features: Feature matrix
            labels: Labels DataFrame
        
        Returns:
            List of experiment results
        """
        print("\\nRunning architecture experiments")
        
        architectures = self.config['experiments']['architectures']
        attributes = self.config['experiments']['attributes']
        
        all_results = []
        
        for arch_config in architectures:
            layers = arch_config['layers']
            node_options = arch_config['nodes']
            
            for nodes in node_options:
                print(f"\\nTesting architecture: {layers} layers, {nodes} nodes")
                
                # Update model config
                model_config = self.config['model'].copy()
                model_config['hidden_layers'] = layers
                model_config['hidden_size'] = nodes
                
                for attribute in attributes:
                    # Find optimal dropout rate for this attribute (from previous experiments)
                    if attribute == 'valence':
                        dropout_rate = 0.7  # From paper results
                    else:
                        dropout_rate = 0.5  # From paper results
                    
                    model_config['dropout_rate'] = dropout_rate
                    
                    # Run experiment
                    results = self.run_single_architecture_experiment(
                        features, labels, attribute, model_config
                    )
                    
                    all_results.extend(results)
        
        return all_results
    
    def run_single_architecture_experiment(
        self,
        features: np.ndarray,
        labels: pd.DataFrame,
        attribute: str,
        model_config: Dict
    ) -> List[Dict[str, float]]:
        """
        Run experiment for single architecture configuration.
        
        Args:
            features: Feature matrix
            labels: Labels DataFrame
            attribute: Emotion attribute
            model_config: Model configuration
        
        Returns:
            List of results
        """
        # Get data splits
        train_speakers, val_speakers, test_speakers = get_paper_data_splits(labels)
        train_indices, val_indices, test_indices = create_speaker_splits(
            labels, train_speakers, val_speakers, test_speakers
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader, scaler = create_data_loaders(
            features, labels, train_indices, val_indices, test_indices,
            attribute=attribute,
            batch_size=self.config['training']['batch_size']
        )
        
        num_runs = self.config['experiments']['num_runs']
        results = []
        
        for run in range(num_runs):
            # Create model
            model = EmotionMLP(
                input_size=self.config['dataset']['feature_dim'],
                hidden_sizes=[model_config['hidden_size']] * model_config['hidden_layers'],
                dropout_rate=model_config['dropout_rate'],
                use_batch_norm=model_config['batch_norm']
            )
            model.to(self.device)
            
            # Train model
            experiment_config = {
                'run': run,
                'num_layers': model_config['hidden_layers'],
                'hidden_size': model_config['hidden_size'],
                'speaker_dependent': False
            }
            
            result = self.train_model(
                model, train_loader, val_loader, test_loader,
                attribute, model_config['dropout_rate'], experiment_config
            )
            
            results.append(result)
        
        return results
    
    def run_speaker_dependency_experiment(
        self,
        features: np.ndarray,
        labels: pd.DataFrame
    ) -> List[Dict[str, float]]:
        """
        Run speaker dependency analysis experiment.
        
        Args:
            features: Feature matrix
            labels: Labels DataFrame
        
        Returns:
            List of experiment results
        """
        print("\\nRunning speaker dependency experiments")
        
        attributes = self.config['experiments']['attributes']
        all_results = []
        
        for attribute in attributes:
            print(f"\\nTesting {attribute}")
            
            # Speaker independent
            print("Speaker independent condition:")
            indep_results = self.run_dropout_experiment(
                features, labels, attribute, speaker_dependent=False
            )
            
            # Speaker dependent  
            print("Speaker dependent condition:")
            dep_results = self.run_dropout_experiment(
                features, labels, attribute, speaker_dependent=True
            )
            
            all_results.extend(indep_results)
            all_results.extend(dep_results)
        
        return all_results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Speech Emotion Recognition Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default='dropout',
                       choices=['dropout', 'architecture', 'speaker_dependency', 'all'],
                       help='Experiment type to run')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (overrides config)')
    parser.add_argument('--use_dummy_data', action='store_true',
                       help='Use dummy data for testing')
    parser.add_argument('--use_esd_data', action='store_true',
                       help='Use ESD dataset with OpenSMILE features')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--dropout_rate', type=float, default=None,
                       help='Specific dropout rate to use (overrides config/experiment)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--validation_frequency', type=int, default=None,
                       help='Validate every N epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config values with command line arguments if provided
    if args.data_dir:
        config['dataset']['data_dir'] = args.data_dir
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
        print(f"Overriding epochs: {args.epochs}")
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        print(f"Overriding batch size: {args.batch_size}")
    
    if args.validation_frequency:
        config['training']['validation_frequency'] = args.validation_frequency
        print(f"Overriding validation frequency: {args.validation_frequency}")
    
    if args.dropout_rate is not None:
        # Override experiment to run single dropout rate
        config['experiments']['dropout_rates'] = [args.dropout_rate]
        config['model']['dropout_rate'] = args.dropout_rate
        print(f"Overriding dropout rate: {args.dropout_rate}")
    
    # Create experiment directory
    exp_dir = create_experiment_dir(
        config['logging']['log_dir'],
        f"ser_regularization_{args.experiment}"
    )
    
    # Initialize trainer
    trainer = Trainer(config, args.device)
    
    # Load data
    print("Loading data...")
    if args.use_dummy_data:
        print("Using dummy data for testing")
        features, labels = create_dummy_data(
            n_samples=2000,
            feature_dim=config['dataset']['feature_dim'],
            n_speakers=50,
            save_path=config['dataset']['data_dir']
        )
    elif hasattr(args, 'use_esd_data') and args.use_esd_data:
        print("Using ESD dataset with OpenSMILE features")
        try:
            features, labels = load_esd_opensmile_data()
            config['dataset']['feature_dim'] = features.shape[1]  # Update config with actual feature dim
        except FileNotFoundError as e:
            print(f"ESD dataset not found: {e}")
            print("Using dummy data instead.")
            features, labels = create_dummy_data(
                n_samples=2000,
                feature_dim=config['dataset']['feature_dim'],
                n_speakers=50,
                save_path=config['dataset']['data_dir']
            )
    else:
        try:
            features, labels = load_msp_podcast_data(config['dataset']['data_dir'])
        except FileNotFoundError:
            print("Dataset not found. Using dummy data instead.")
            features, labels = create_dummy_data(
                n_samples=2000,
                feature_dim=config['dataset']['feature_dim'], 
                n_speakers=50,
                save_path=config['dataset']['data_dir']
            )
    
    # Run experiments
    all_results = []
    
    if args.experiment == 'dropout' or args.experiment == 'all':
        print("=== DROPOUT RATE EXPERIMENTS ===")
        for attribute in config['experiments']['attributes']:
            results = trainer.run_dropout_experiment(features, labels, attribute)
            all_results.extend(results)
    
    if args.experiment == 'architecture' or args.experiment == 'all':
        print("\\n=== ARCHITECTURE EXPERIMENTS ===")
        arch_results = trainer.run_architecture_experiment(features, labels)
        all_results.extend(arch_results)
    
    if args.experiment == 'speaker_dependency' or args.experiment == 'all':
        print("\\n=== SPEAKER DEPENDENCY EXPERIMENTS ===")
        speaker_results = trainer.run_speaker_dependency_experiment(features, labels)
        all_results.extend(speaker_results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(exp_dir, 'all_results.csv'), index=False)
    
    # Analysis and plotting
    if len(all_results) > 0:
        print("\\n=== RESULTS ANALYSIS ===")
        
        # Find optimal dropout rates
        optimal_rates = analyze_optimal_dropout_rates(results_df)
        print("Optimal dropout rates:")
        for attr, rate in optimal_rates.items():
            print(f"  {attr}: {rate}")
        
        # Save experiment summary
        save_experiment_summary(results_df, config, exp_dir)
        
        print(f"\\nExperiment completed. Results saved to: {exp_dir}")
    
    else:
        print("No results to analyze.")


if __name__ == '__main__':
    main()