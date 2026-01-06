# Complete Training Example - Speech Emotion Recognition

This directory contains a **complete working example** of the speech emotion recognition pipeline with OpenSMILE features and enhanced ESD dataset.

## What This Example Demonstrates

- **Complete ML Pipeline**: Data loading → Training → Evaluation → Visualization
- **Production Features**: 6,373 OpenSMILE ComParE features (vs basic librosa features)
- **Real Dataset**: 2,500 ESD audio samples with dimensional emotion labels
- **Advanced Training**: 100-epoch training with validation scheduling and early stopping
- **Professional Plots**: Training/validation curves with publication-quality formatting
- **Reproducible Results**: Saved models, configs, and detailed logging

## Quick Start

### Prerequisites
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract OpenSMILE features (one-time setup)
python src/extract_opensmile_features.py --use_unified_dataset

# 3. Verify data loading works
python test_esd_training.py
```

### Run Complete Example
```bash
# Single command for full demonstration
python run_example_training.py
```

**What this does:**
- Trains neural networks for valence prediction
- Tests 4 dropout rates: 0.0, 0.3, 0.5, 0.7
- Runs 100 epochs with validation every 5 epochs
- Generates training curves automatically
- Saves everything to `experiments/` directory

**Expected output:**
- Training time: ~10-15 minutes on CPU
- Results: 4 experiment folders with plots and data
- File size: ~50MB total

## Generated Results

After running, you'll have:

```
experiments/
├── valence_dropout_0.0/
│ ├── valence_dropout_0.0_curves.png # Training curves
│ └── [model files and logs]
├── valence_dropout_0.3/
│ ├── valence_dropout_0.3_curves.png
│ └── [model files and logs]
├── valence_dropout_0.5/
│ ├── valence_dropout_0.5_curves.png
│ └── [model files and logs]
└── valence_dropout_0.7/
├── valence_dropout_0.7_curves.png
└── [model files and logs]
```

## Training Curve Examples

Each PNG file contains:
- **Left plot**: Training/Validation Loss (1-CCC) over epochs
- **Right plot**: Training/Validation CCC over epochs 
- **Orange line**: Early stopping point (best model)
- **Red dot**: Best validation performance
- **Markers**: Validation points (every 5 epochs)

## Configuration Details

**Model Architecture:**
- Input: 6,373 OpenSMILE ComParE features
- Hidden: 2 layers × 256 neurons
- Dropout: Variable (0.0, 0.3, 0.5, 0.7)
- Output: Single valence value
- Loss: Concordance Correlation Coefficient (CCC)

**Training Setup:**
- Optimizer: SGD with momentum (0.9)
- Learning Rate: 0.001
- Batch Size: 16 (CPU optimized)
- Validation: Every 5 epochs
- Early Stopping: Patience 20 epochs
- Total Epochs: 100

**Data Splits:**
- Train: 1,250 samples (50%)
- Validation: 500 samples (20%) 
- Test: 750 samples (30%)
- Split Type: Speaker-independent

## Customizing the Example

### Different Emotions
Edit `config/cpu_example_config.yaml`:
```yaml
experiments:
attributes: ["arousal"] # or ["dominance"] or ["valence", "arousal", "dominance"]
```

### More Dropout Rates
```yaml
experiments:
dropout_rates: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

### GPU Training
```bash
python run_example_training.py --device cuda
```

### Longer Training
```yaml
training:
epochs: 200
validation_frequency: 10
```

## Expected Results

**Typical Performance (Valence Prediction):**
- **Baseline (0.0 dropout)**: CCC ~0.3-0.4
- **Optimal dropout**: CCC ~0.4-0.5 
- **Overfitting (high dropout)**: CCC ~0.2-0.3

**Training Curves Should Show:**
- Decreasing loss over epochs
- Increasing CCC over epochs
- Validation plateau (early stopping)
- Clear overfitting prevention with optimal dropout

## 🐛 Troubleshooting

**"ESD dataset not found":**
```bash
# Make sure OpenSMILE features are extracted first
python src/extract_opensmile_features.py --use_unified_dataset
```

**"Out of memory":**
- Reduce batch size in config: `batch_size: 8`
- Use fewer dropout rates for testing

**"Training too slow":**
- Reduce epochs: `epochs: 50`
- Increase validation frequency: `validation_frequency: 10`

## Paper Reference

This implementation reproduces:
> Sridhar et al. (2018) "Role of Regularization in the Prediction of Valence from Speech"

**Key Enhancements:**
- Real ESD dataset (vs MSP-Podcast)
- OpenSMILE ComParE features (vs basic features)
- Dimensional labels from wav2vec 2.0 (vs categorical only)
- Enhanced training pipeline with validation scheduling
- Professional plotting and result visualization

## Use as GitHub Example

This example serves as:
- **Working demonstration** of the full pipeline
- **Reference implementation** for researchers
- **Reproducible baseline** for comparisons
- **Tutorial** for using the codebase

Perfect for showcasing your research implementation! 