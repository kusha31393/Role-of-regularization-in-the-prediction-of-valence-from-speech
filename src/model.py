"""
Neural network models for speech emotion recognition.
Implements the DNN architecture described in the paper with configurable dropout.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np


class EmotionMLP(nn.Module):
    """
    Multi-Layer Perceptron for emotion regression.
    
    Architecture matches the paper:
    - ReLU activation for hidden layers
    - Linear activation for output layer  
    - Batch normalization for hidden layers
    - Configurable dropout rates
    - Single output for regression
    """
    
    def __init__(
        self,
        input_size: int = 6373,  # OpenSmile ComParE 2013 features
        hidden_sizes: List[int] = [256, 256],
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize the MLP model.
        
        Args:
            input_size: Dimension of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout probability for hidden layers
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(EmotionMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build the network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input to first hidden layer
        layer_sizes = [input_size] + hidden_sizes
        
        for i in range(len(hidden_sizes)):
            # Linear layer
            linear_layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(linear_layer)
            
            # Batch normalization (if enabled)
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            
            # Dropout layer
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer (no dropout, no batch norm)
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        
        # Activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Process through hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply batch normalization
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout
            x = self.dropouts[i](x)
        
        # Output layer (linear activation)
        x = self.output_layer(x)
        
        return x.squeeze(-1)  # Remove last dimension for regression
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model_from_config(config: dict) -> EmotionMLP:
    """
    Create model from configuration dictionary.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Initialized EmotionMLP model
    """
    model_config = config.get('model', {})
    
    # Handle different ways of specifying hidden layers
    hidden_layers = model_config.get('hidden_layers', 2)
    hidden_size = model_config.get('hidden_size', 256)
    
    if isinstance(hidden_layers, int):
        hidden_sizes = [hidden_size] * hidden_layers
    elif isinstance(hidden_layers, list):
        hidden_sizes = hidden_layers
    else:
        hidden_sizes = [256, 256]  # Default
    
    model = EmotionMLP(
        input_size=config.get('dataset', {}).get('feature_dim', 6373),
        hidden_sizes=hidden_sizes,
        dropout_rate=model_config.get('dropout_rate', 0.5),
        use_batch_norm=model_config.get('batch_norm', True),
        activation=model_config.get('activation', 'relu')
    )
    
    return model


class MultiTaskMLP(nn.Module):
    """
    Multi-task learning model for predicting multiple emotion attributes.
    Can be used for joint training of valence, arousal, and dominance.
    """
    
    def __init__(
        self,
        input_size: int = 6373,
        hidden_sizes: List[int] = [256, 256],
        num_tasks: int = 3,  # valence, arousal, dominance
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        super(MultiTaskMLP, self).__init__()
        
        # Shared hidden layers
        self.shared_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        layer_sizes = [input_size] + hidden_sizes
        
        for i in range(len(hidden_sizes)):
            self.shared_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Task-specific output layers
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[-1], 1) for _ in range(num_tasks)
        ])
        
        # Activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.shared_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        for layer in self.output_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            List of output tensors, one for each task
        """
        # Process through shared hidden layers
        for i, layer in enumerate(self.shared_layers):
            x = layer(x)
            
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropouts[i](x)
        
        # Task-specific outputs
        outputs = []
        for output_layer in self.output_layers:
            task_output = output_layer(x)
            outputs.append(task_output.squeeze(-1))
        
        return outputs


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model information
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'num_parameters': num_params,
        'model_type': type(model).__name__,
    }
    
    if hasattr(model, 'hidden_sizes'):
        info['hidden_sizes'] = model.hidden_sizes
        info['dropout_rate'] = model.dropout_rate
    
    return info