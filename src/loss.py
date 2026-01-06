"""
Loss functions for speech emotion recognition.
Implements Concordance Correlation Coefficient (CCC) as used in the paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient Loss
    
    CCC measures both correlation and agreement between predictions and targets.
    CCC = (2 * pearson * std_pred * std_true) / (var_pred + var_true + (mean_pred - mean_true)^2)
    
    This is the primary loss function used in the paper.
    """
    
    def __init__(self, reduction='mean', epsilon=1e-8):
        super(CCCLoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate CCC loss between predictions and targets.
        
        Args:
            predictions: Predicted values, shape (batch_size,)
            targets: Ground truth values, shape (batch_size,)
        
        Returns:
            CCC loss (1 - CCC to make it a loss function)
        """
        # Ensure tensors are 1D
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate means
        mean_pred = torch.mean(predictions)
        mean_true = torch.mean(targets)
        
        # Center the variables
        pred_centered = predictions - mean_pred
        true_centered = targets - mean_true
        
        # Calculate variances and covariance
        var_pred = torch.mean(pred_centered ** 2)
        var_true = torch.mean(true_centered ** 2)
        covariance = torch.mean(pred_centered * true_centered)
        
        # Calculate CCC
        numerator = 2 * covariance
        denominator = var_pred + var_true + (mean_pred - mean_true) ** 2 + self.epsilon
        ccc = numerator / denominator
        
        # Return 1 - CCC as loss (to minimize)
        loss = 1 - ccc
        
        return loss


def calculate_ccc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate CCC for evaluation (not as loss).
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        CCC value as float
    """
    with torch.no_grad():
        ccc_loss = CCCLoss()
        loss = ccc_loss(predictions, targets)
        return 1 - loss.item()


def calculate_pearson_correlation(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        Pearson correlation as float
    """
    with torch.no_grad():
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        mean_pred = torch.mean(predictions)
        mean_true = torch.mean(targets)
        
        pred_centered = predictions - mean_pred
        true_centered = targets - mean_true
        
        numerator = torch.sum(pred_centered * true_centered)
        denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(true_centered ** 2))
        
        if denominator > 0:
            correlation = numerator / denominator
            return correlation.item()
        else:
            return 0.0


class MSELoss(nn.Module):
    """Mean Squared Error Loss for comparison."""
    
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mse(predictions, targets)


def get_loss_function(loss_type: str) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: Type of loss function ('ccc', 'mse')
    
    Returns:
        Loss function module
    """
    if loss_type.lower() == 'ccc':
        return CCCLoss()
    elif loss_type.lower() == 'mse':
        return MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def evaluate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Calculate multiple evaluation metrics.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
    
    Returns:
        Dictionary with CCC, Pearson correlation, and MSE
    """
    with torch.no_grad():
        ccc = calculate_ccc(predictions, targets)
        pearson = calculate_pearson_correlation(predictions, targets)
        mse = nn.MSELoss()(predictions, targets).item()
        
        return {
            'ccc': ccc,
            'pearson': pearson,
            'mse': mse
        }