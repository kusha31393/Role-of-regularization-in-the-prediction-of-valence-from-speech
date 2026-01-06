# Training Results and Experiments

This directory contains all training results, model checkpoints, and experimental outputs from the speech emotion recognition pipeline.

## **Directory Structure**

```
results/
├── model_architecture.png # Neural network architecture diagram
├── experiments/ # Training experiment results
│ ├── valence_dropout_0.0/ # No dropout baseline
│ │ └── best_model_dropout_0.0.pth
│ └── valence_dropout_0.5/ # Optimal dropout example
│ ├── best_model_dropout_0.5.pth # Best model checkpoint
│ ├── final_results_dropout_0.5.pth # Complete training data
│ └── valence_dropout_0.5_curves.png # Training curves
└── [future experiments...]
```

## **Available Results**

### **Valence Prediction with Dropout=0.5**
**Location**: `experiments/valence_dropout_0.5/`

**Performance:**
- **Best Validation CCC**: 0.1084 (epoch 18)
- **Final Test CCC**: 0.0748
- **Training Epochs**: 20 epochs, validation every 2 epochs
- **Dataset**: 2,500 ESD samples with 6,373 OpenSMILE features

**Files:**
- `best_model_dropout_0.5.pth` (13.6MB) - Trained model weights and optimizer state
- `final_results_dropout_0.5.pth` (6.8MB) - Complete training history and metrics
- `valence_dropout_0.5_curves.png` (334KB) - Training and validation curves

### **Valence Prediction with Dropout=0.0** 
**Location**: `experiments/valence_dropout_0.0/`

**Purpose**: Baseline comparison without regularization
- Demonstrates overfitting without dropout
- Lower validation performance expected
- Reference for dropout effectiveness

## **Loading and Analyzing Results**

### **Load Checkpoint Data**
```python
import torch
from src.train import load_checkpoint
from src.model import EmotionMLP

# Create model instance
model = EmotionMLP(input_size=6373, hidden_sizes=[256, 256], dropout_rate=0.5)

# Load checkpoint
checkpoint = load_checkpoint('results/experiments/valence_dropout_0.5/best_model_dropout_0.5.pth', model)

# Access training data
train_ccc_history = checkpoint['train_ccc_history']
val_ccc_history = checkpoint['val_ccc_history']
best_val_ccc = checkpoint['best_val_ccc']
```

### **Load Complete Results**
```python
import torch

# Load final results
results = torch.load('results/experiments/valence_dropout_0.5/final_results_dropout_0.5.pth', 
map_location='cpu')

print(f"Training completed in {results['training_time_minutes']:.1f} minutes")
print(f"Best validation CCC: {results['best_val_ccc']:.4f}")
print(f"Final test performance:")
print(f" CCC: {results['final_test_ccc']:.4f}")
print(f" Pearson: {results['final_test_pearson']:.4f}")
print(f" MSE: {results['final_test_mse']:.4f}")
```

### **Visualize Training Progress**
```python
import matplotlib.pyplot as plt

# Load and plot training curves
results = torch.load('results/experiments/valence_dropout_0.5/final_results_dropout_0.5.pth')

train_history = results['train_ccc_history']
val_history = results['val_ccc_history']
validation_epochs = results['validation_epochs']

plt.figure(figsize=(10, 6))
plt.plot(validation_epochs, train_history, 'b-', label='Training CCC', marker='o')
plt.plot(validation_epochs, val_history, 'r-', label='Validation CCC', marker='s')
plt.xlabel('Epoch')
plt.ylabel('CCC')
plt.title('Training Progress - Valence Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## **Result Interpretation**

### **Training Curves Analysis**
The generated plots show:
- **Blue line**: Training CCC (should increase over epochs)
- **Red line**: Validation CCC (should increase then plateau)
- **Orange vertical line**: Best model epoch (early stopping point)
- **Red dot**: Best validation performance marker

### **Performance Expectations**
**For valence prediction:**
- **CCC > 0.1**: Good performance (valence is challenging)
- **Training > Validation**: Normal (some overfitting expected)
- **Early stopping**: Prevents severe overfitting
- **Test CCC**: Should be close to validation CCC

### **Checkpoint Contents**
Each checkpoint contains:
- **Model weights**: Trained neural network parameters
- **Optimizer state**: For resuming training
- **Training history**: Complete loss/CCC curves
- **Configuration**: All hyperparameters used
- **Metadata**: Epoch, dropout rate, performance metrics

## **Using Results for Research**

### **Publication Figures**
- Training curves are publication-ready (300 DPI)
- Use for comparing different regularization approaches
- Demonstrate learning convergence and overfitting prevention

### **Baseline Comparisons**
- Use saved models as baselines for new approaches
- Compare performance against these established results
- Extend to other emotions (arousal, dominance)

### **Reproducibility**
- All results include complete configuration
- Checkpoints enable exact reproduction
- Random seeds ensure deterministic results
- Training curves validate implementation correctness

## **Expected Future Results**

As you run more experiments, this directory will contain:
- **Multiple dropout rates**: 0.0, 0.1, 0.2, ..., 0.9
- **Different emotions**: valence, arousal, dominance results
- **Architecture variations**: 2-layer, 4-layer, 6-layer networks
- **Speaker dependency**: Independent vs dependent training results

## **File Size Management**

**Checkpoint sizes:**
- **Model weights**: ~13MB per checkpoint (3.4M parameters × 4 bytes)
- **Training history**: ~7MB per experiment (includes all epoch data)
- **Plots**: ~300KB per figure (high-resolution PNG)

**Storage tips:**
- Keep only best checkpoints for each configuration
- Compress historical results for long-term storage
- Use `.gitignore` to avoid committing large files accidentally