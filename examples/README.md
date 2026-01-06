# Examples and Testing

This directory contains example scripts and tests to demonstrate the speech emotion recognition pipeline.

## **Available Examples**

### **Training Examples**

#### `run_example_training.py`
**Complete training demonstration** - The main example for GitHub
- Trains neural network for valence prediction
- Uses dropout rate 0.5 for 20 epochs
- Generates training curves and saves checkpoints
- Perfect for demonstrating the full pipeline

```bash
python examples/run_example_training.py
```

**Outputs:**
- `results/experiments/valence_dropout_0.5/valence_dropout_0.5_curves.png`
- `results/experiments/valence_dropout_0.5/best_model_dropout_0.5.pth`
- `results/experiments/valence_dropout_0.5/final_results_dropout_0.5.pth`

#### `load_checkpoint_example.py` 
**Checkpoint loading and analysis** - Shows how to load saved models
- Lists available checkpoints
- Loads trained model weights
- Analyzes training history
- Demonstrates model inference

```bash
python examples/load_checkpoint_example.py
```

### **Testing Scripts**

#### `test_esd_training.py`
**Basic pipeline verification** - Quick test to ensure everything works
- Tests ESD dataset loading (2,500 samples)
- Verifies OpenSMILE feature dimensions (6,373 features)
- Tests model creation and forward pass
- Validates loss computation

```bash
python examples/test_esd_training.py
```

#### `test_loss_curves.py`
**Loss curve plotting verification** - Tests visualization functionality
- Creates mock training data
- Tests individual curve plotting
- Tests combined loss curve plotting
- Verifies plot saving functionality

```bash
python examples/test_loss_curves.py
```

## **Getting Started**

### **1. Quick Verification**
```bash
# Test that everything is set up correctly
python examples/test_esd_training.py
```

### **2. Run Complete Example**
```bash
# Generate training curves and results
python examples/run_example_training.py
```

### **3. Analyze Results**
```bash
# Load and examine the trained model
python examples/load_checkpoint_example.py
```

## **Customizing Examples**

### **Different Training Configurations**

**Longer training:**
```bash
# Edit run_example_training.py and change:
'--epochs', '100', # Instead of '20'
'--validation_frequency', '10' # Instead of '2'
```

**Different dropout rates:**
```bash
# Edit the sys.argv in run_example_training.py:
'--dropout_rate', '0.3', # Instead of '0.5'
```

**GPU training:**
```bash
# Edit device in run_example_training.py:
device = 'cuda' # Instead of 'cpu'
```

### **Testing Different Components**

**Test individual modules:**
```bash
# Test data loading only
python -c "from src.data_loader import load_esd_opensmile_data; features, labels = load_esd_opensmile_data(); print(f'Loaded: {features.shape}')"

# Test model creation only 
python -c "from src.model import EmotionMLP; model = EmotionMLP(input_size=6373); print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')"

# Test loss function only
python -c "from src.loss import CCCLoss; import torch; loss_fn = CCCLoss(); print('CCC loss function created')"
```

## **Expected Results**

### **test_esd_training.py**
```
Data loaded successfully!
Features shape: (2500, 6373)
Labels shape: (2500, 4)
Model created successfully!
Parameters: 3,396,609
Loss computation successful: 0.9930
All tests passed!
```

### **run_example_training.py**
```
TRAINING COMPLETED SUCCESSFULLY!
Best validation CCC: 0.1084 (epoch 18)
Final test CCC: 0.0748
Generated files in results/experiments/valence_dropout_0.5/
```

### **test_loss_curves.py**
```
Individual CCC curve plotting successful!
Combined loss curve plotting successful!
Check generated files: test_ccc_curves.png, test_combined_curves.png
```

## 🐛 **Troubleshooting**

### Common Issues

**"ESD dataset not found"**
```bash
# Make sure OpenSMILE features are extracted first
python src/extract_opensmile_features.py --use_unified_dataset
```

**"OpenSMILE not found"**
```bash
# Install OpenSMILE
pip install opensmile
```

**"Out of memory"**
```bash
# Use CPU instead of GPU
# Edit device = 'cpu' in the example scripts
```

### **Performance Issues**

**Training too slow:**
- Reduce epochs in example scripts
- Increase validation frequency
- Use smaller batch sizes

**Large file sizes:**
- Checkpoints are ~13MB (normal for 6,373 input features)
- Results files are ~7MB (contains full training history)
- PNG plots are ~300KB (high-resolution)

## **Next Steps**

After running the examples:
1. **Explore results** in `results/experiments/`
2. **Analyze training curves** using the generated plots
3. **Load checkpoints** to understand model performance
4. **Experiment** with different configurations
5. **Extend** to your own datasets or research questions

## **Further Reading**

- **[Main README](../README.md)** - Complete project overview
- **[Code Review](../docs/CODE_REVIEW_ANALYSIS.md)** - Implementation analysis
- **[Training Guide](../docs/EXAMPLE_TRAINING_README.md)** - Detailed tutorial
- **[Source Code](../src/)** - Core implementation with documentation