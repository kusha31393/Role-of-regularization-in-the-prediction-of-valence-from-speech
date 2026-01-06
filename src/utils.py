"""
Utility functions for experiments, logging, and analysis.
"""

import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import random
from datetime import datetime
import pickle


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def create_experiment_dir(base_dir: str, experiment_name: str = None) -> str:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base experiments directory
        experiment_name: Optional experiment name
    
    Returns:
        Path to created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        exp_dir = f"{experiment_name}_{timestamp}"
    else:
        exp_dir = f"experiment_{timestamp}"
    
    exp_path = Path(base_dir) / exp_dir
    exp_path.mkdir(parents=True, exist_ok=True)
    
    return str(exp_path)


class ExperimentLogger:
    """
    Logger for experiment results and metrics.
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Initialize log files
        self.metrics_file = self.log_dir / "metrics.json"
        self.results_file = self.log_dir / "results.csv"
        
        # Initialize metrics storage
        self.metrics = {}
        self.results = []
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics for a training step.
        
        Args:
            metrics: Dictionary of metric values
            step: Training step number
        """
        entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Save to metrics file
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        all_metrics.append(entry)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def log_experiment_result(self, config: Dict[str, Any], results: Dict[str, float]):
        """
        Log results from a complete experiment run.
        
        Args:
            config: Experiment configuration
            results: Final results
        """
        entry = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            **config,
            **results
        }
        
        self.results.append(entry)
        
        # Save to CSV
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
    
    def save_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, filename: str = None):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            filename: Optional filename
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, self.log_dir / filename)
    
    def load_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   filename: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer  
            filename: Checkpoint filename
        
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(self.log_dir / filename)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch']


def plot_dropout_analysis(results_df: pd.DataFrame, save_path: str = None):
    """
    Plot dropout rate analysis results similar to Figure 1 in the paper.
    
    Args:
        results_df: DataFrame with columns ['dropout_rate', 'attribute', 'ccc_val', 'ccc_test']
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    attributes = ['valence', 'arousal', 'dominance']
    sets = ['validation', 'test']
    
    for i, attr in enumerate(attributes):
        for j, data_set in enumerate(sets):
            ax = axes[j, i]
            
            # Filter data for this attribute
            attr_data = results_df[results_df['attribute'] == attr]
            
            # Plot CCC vs dropout rate
            ccc_col = f'ccc_{data_set[:3]}'  # 'ccc_val' or 'ccc_tes'
            if ccc_col in attr_data.columns:
                ax.plot(attr_data['dropout_rate'], attr_data[ccc_col], 'o-', 
                       linewidth=2, markersize=6)
            
            ax.set_xlabel('Dropout Rate')
            ax.set_ylabel('CCC')
            ax.set_title(f'{attr.title()} - {data_set.title()} Set')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_architecture_comparison(results_df: pd.DataFrame, save_path: str = None):
    """
    Plot comparison of different network architectures.
    
    Args:
        results_df: Results DataFrame
        save_path: Optional save path
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    attributes = ['valence', 'arousal', 'dominance']
    
    for i, attr in enumerate(attributes):
        ax = axes[i]
        attr_data = results_df[results_df['attribute'] == attr]
        
        # Group by architecture parameters
        if 'num_layers' in attr_data.columns and 'hidden_size' in attr_data.columns:
            grouped = attr_data.groupby(['num_layers', 'hidden_size'])
            
            for (layers, size), group in grouped:
                ax.plot(group['dropout_rate'], group['ccc_val'], 
                       label=f'{layers}L-{size}N', marker='o')
        
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('CCC (Validation)')
        ax.set_title(f'{attr.title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_speaker_dependency_analysis(results_df: pd.DataFrame, save_path: str = None):
    """
    Plot speaker dependency analysis (Table 2 from paper).
    
    Args:
        results_df: DataFrame with speaker-dependent vs independent results
        save_path: Optional save path
    """
    # Calculate relative improvements
    comparison_data = []
    
    attributes = ['valence', 'arousal', 'dominance']
    
    for attr in attributes:
        attr_data = results_df[results_df['attribute'] == attr]
        
        if 'speaker_dependent' in attr_data.columns:
            indep_ccc = attr_data[attr_data['speaker_dependent'] == False]['ccc_test'].mean()
            dep_ccc = attr_data[attr_data['speaker_dependent'] == True]['ccc_test'].mean()
            
            relative_gain = ((dep_ccc - indep_ccc) / indep_ccc) * 100
            
            comparison_data.append({
                'attribute': attr,
                'speaker_independent': indep_ccc,
                'speaker_dependent': dep_ccc,
                'relative_gain': relative_gain
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # CCC comparison
    x = np.arange(len(attributes))
    width = 0.35
    
    ax1.bar(x - width/2, comparison_df['speaker_independent'], width, 
            label='Speaker Independent', alpha=0.8)
    ax1.bar(x + width/2, comparison_df['speaker_dependent'], width,
            label='Speaker Dependent', alpha=0.8)
    
    ax1.set_xlabel('Emotion Attribute')
    ax1.set_ylabel('CCC (Test Set)')
    ax1.set_title('Speaker Independent vs Dependent Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels([attr.title() for attr in attributes])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Relative gain
    colors = ['red' if gain > 20 else 'blue' for gain in comparison_df['relative_gain']]
    bars = ax2.bar(attributes, comparison_df['relative_gain'], color=colors, alpha=0.7)
    
    ax2.set_xlabel('Emotion Attribute')
    ax2.set_ylabel('Relative Gain (%)')
    ax2.set_title('Relative Improvement with Speaker-Dependent Training')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, gain in zip(bars, comparison_df['relative_gain']):
        height = bar.get_height()
        ax2.annotate(f'{gain:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(train_history: List[float], val_history: List[float], 
                         metric_name: str = 'CCC', save_path: str = None, 
                         title: str = None, early_stop_epoch: int = None):
    """
    Plot training and validation curves over epochs.
    
    Args:
        train_history: List of training metric values per epoch
        val_history: List of validation metric values per epoch  
        metric_name: Name of the metric being plotted (e.g., 'CCC', 'Loss')
        save_path: Path to save the plot
        title: Custom title for the plot
        early_stop_epoch: Epoch where early stopping occurred (if any)
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_history) + 1)
    
    plt.plot(epochs, train_history, 'b-', label=f'Training {metric_name}', linewidth=2)
    plt.plot(epochs, val_history, 'r-', label=f'Validation {metric_name}', linewidth=2)
    
    # Mark early stopping point
    if early_stop_epoch is not None:
        plt.axvline(x=early_stop_epoch, color='orange', linestyle='--', 
                   label=f'Early Stop (Epoch {early_stop_epoch})', alpha=0.7)
    
    # Mark best validation performance
    if val_history:
        if metric_name.lower() in ['ccc', 'pearson', 'accuracy']:  # Higher is better
            best_val_idx = np.argmax(val_history)
            best_val_value = max(val_history)
        else:  # Lower is better (loss, mse, etc.)
            best_val_idx = np.argmin(val_history)
            best_val_value = min(val_history)
        
        plt.plot(best_val_idx + 1, best_val_value, 'ro', markersize=8, 
                label=f'Best Val {metric_name}: {best_val_value:.4f}')
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(title or f'Training and Validation {metric_name} Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add final values as text
    if train_history and val_history:
        final_train = train_history[-1]
        final_val = val_history[-1]
        plt.text(0.02, 0.98, f'Final Train {metric_name}: {final_train:.4f}\nFinal Val {metric_name}: {final_val:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to {save_path}")
    
    plt.show()


def plot_loss_curves(train_losses: List[float], val_losses: List[float], 
                     train_cccs: List[float], val_cccs: List[float], 
                     save_path: str = None, early_stop_epoch: int = None):
    """
    Plot both loss and CCC curves in a combined view.
    
    Args:
        train_losses: Training loss values per epoch
        val_losses: Validation loss values per epoch
        train_cccs: Training CCC values per epoch  
        val_cccs: Validation CCC values per epoch
        save_path: Path to save the plot
        early_stop_epoch: Epoch where early stopping occurred
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot Loss (1 - CCC)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    if early_stop_epoch is not None:
        ax1.axvline(x=early_stop_epoch, color='orange', linestyle='--', 
                   label=f'Early Stop (Epoch {early_stop_epoch})', alpha=0.7)
    
    # Mark best validation loss
    if val_losses:
        best_val_loss_idx = np.argmin(val_losses)
        best_val_loss = min(val_losses)
        ax1.plot(best_val_loss_idx + 1, best_val_loss, 'ro', markersize=8,
                label=f'Best Val Loss: {best_val_loss:.4f}')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (1 - CCC)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot CCC
    ax2.plot(epochs, train_cccs, 'b-', label='Training CCC', linewidth=2)
    ax2.plot(epochs, val_cccs, 'r-', label='Validation CCC', linewidth=2)
    
    if early_stop_epoch is not None:
        ax2.axvline(x=early_stop_epoch, color='orange', linestyle='--', 
                   label=f'Early Stop (Epoch {early_stop_epoch})', alpha=0.7)
    
    # Mark best validation CCC
    if val_cccs:
        best_val_ccc_idx = np.argmax(val_cccs)
        best_val_ccc = max(val_cccs)
        ax2.plot(best_val_ccc_idx + 1, best_val_ccc, 'ro', markersize=8,
                label=f'Best Val CCC: {best_val_ccc:.4f}')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('CCC')
    ax2.set_title('Training and Validation CCC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to {save_path}")
    
    plt.show()


def statistical_significance_test(results1: List[float], results2: List[float]) -> Tuple[float, bool]:
    """
    Perform one-tailed t-test for statistical significance.
    
    Args:
        results1: First set of results
        results2: Second set of results
    
    Returns:
        Tuple of (p_value, is_significant)
    """
    from scipy import stats
    
    # One-tailed t-test
    t_stat, p_value = stats.ttest_rel(results2, results1, alternative='greater')
    is_significant = p_value < 0.001  # p < 0.001 as used in paper
    
    return p_value, is_significant


def analyze_optimal_dropout_rates(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Find optimal dropout rates for each emotion attribute.
    
    Args:
        results_df: Results DataFrame
    
    Returns:
        Dictionary with optimal dropout rates
    """
    optimal_rates = {}
    
    attributes = ['valence', 'arousal', 'dominance']
    
    for attr in attributes:
        attr_data = results_df[results_df['attribute'] == attr]
        
        # Find dropout rate with maximum validation CCC
        best_idx = attr_data['ccc_val'].idxmax()
        optimal_rate = attr_data.loc[best_idx, 'dropout_rate']
        optimal_rates[attr] = optimal_rate
    
    return optimal_rates


def save_experiment_summary(results_df: pd.DataFrame, config: Dict[str, Any], 
                          save_dir: str):
    """
    Save comprehensive experiment summary.
    
    Args:
        results_df: Results DataFrame
        config: Experiment configuration
        save_dir: Directory to save summary
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    results_df.to_csv(save_path / "all_results.csv", index=False)
    
    # Find optimal dropout rates
    optimal_rates = analyze_optimal_dropout_rates(results_df)
    
    # Create summary report
    summary = {
        'experiment_config': config,
        'optimal_dropout_rates': optimal_rates,
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(results_df),
        'summary_statistics': results_df.groupby('attribute').agg({
            'ccc_val': ['mean', 'std', 'max'],
            'ccc_test': ['mean', 'std', 'max']
        }).to_dict()
    }
    
    # Save summary
    with open(save_path / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Generate plots
    plot_dropout_analysis(results_df, save_path / "dropout_analysis.png")
    
    print(f"Experiment summary saved to {save_dir}")
    print(f"Optimal dropout rates: {optimal_rates}")