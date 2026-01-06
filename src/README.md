# Source Code Documentation

This directory contains the core implementation of the speech emotion recognition system with dropout regularization.

## **Module Overview**

### **Core Training Pipeline**

#### `train.py` - Main Training Script
**Primary training pipeline** for all experiments
- **Dropout rate experiments**: Tests multiple dropout values (0.0-0.9)
- **Architecture experiments**: Compares different network architectures 
- **Speaker dependency**: Evaluates speaker-dependent vs independent training
- **Checkpointing**: Automatic model saving based on validation performance
- **Visualization**: Generates training curves and performance plots

**Key Functions:**
- `Trainer.train_single_model()` - Core training loop with early stopping
- `Trainer.run_dropout_experiment()` - Dropout rate analysis
- `load_checkpoint()` - Load saved model checkpoints
- `list_available_checkpoints()` - Find available model files

**Usage:**
```bash
# Basic usage
python src/train.py --use_esd_data --experiment dropout --device cpu

# With custom parameters
python src/train.py --use_esd_data --experiment dropout --device cpu \
--dropout_rate 0.5 --epochs 100 --validation_frequency 5
```

### **Neural Network Architecture**

#### `model.py` - Model Definitions
**Neural network architectures** for emotion recognition
- `EmotionMLP` - Multi-layer perceptron for single emotion prediction
- `MultiTaskEmotionMLP` - Multi-task model for V-A-D prediction
- `create_model_from_config()` - Factory function for model creation

**Architecture Details:**
- **Input**: 6,373-dimensional OpenSMILE features
- **Hidden layers**: Configurable (default: 2 layers × 256 neurons)
- **Activation**: ReLU with batch normalization
- **Regularization**: Dropout (configurable rate)
- **Output**: Single emotion dimension or multi-task

### **Loss Functions and Evaluation**

#### `loss.py` - Loss Functions and Metrics
**Concordance Correlation Coefficient (CCC)** implementation
- `CCCLoss` - Primary loss function (1 - CCC)
- `calculate_ccc()` - CCC metric computation
- `evaluate_metrics()` - Comprehensive evaluation (CCC, Pearson, MSE)
- `get_loss_function()` - Loss function factory

**CCC Formula:**
```
CCC = (2 * ρ * σ_x * σ_y) / (σ_x² + σ_y² + (μ_x - μ_y)²)
Loss = 1 - CCC # Minimized during training
```

### **Data Handling**

#### `data_loader.py` - Data Loading and Preprocessing
**Comprehensive data pipeline** for multiple datasets
- `load_esd_opensmile_data()` - Load ESD dataset with OpenSMILE features
- `MSPPodcastDataset` - PyTorch dataset class
- `create_data_loaders()` - Create train/val/test data loaders
- `create_speaker_splits()` - Speaker-aware data splitting
- `FeatureStandardizer` - Feature normalization with outlier clipping

**Data Flow:**
```
Audio Files → OpenSMILE Features → Speaker Splits → PyTorch DataLoaders
```

### **Feature Extraction Pipeline**

#### `extract_opensmile_features.py` - Professional Acoustic Features
**OpenSMILE ComParE feature extraction** from audio files
- Extracts 6,373 ComParE 2016 functional features
- Processes ESD dataset in unified dataset order
- Creates both complete dataset and features-only files
- Comprehensive error handling and metadata tracking

**Usage:**
```bash
# Extract features for unified dataset
python src/extract_opensmile_features.py --use_unified_dataset

# Extract for specific speakers 
python src/extract_opensmile_features.py --speakers "0011,0012" --max_files_per_emotion 25
```

#### `extract_w2v2_emotions.py` - Dimensional Label Extraction
**wav2vec 2.0 dimensional emotion extraction** from audio files
- Downloads and caches pre-trained emotion model (~500MB)
- Extracts arousal, valence, dominance scores from audio
- Creates unified dataset with categorical and dimensional labels
- Processes 2,500 English ESD samples

#### `preprocess_esd.py` - ESD Dataset Preprocessing 
**Enhanced preprocessing pipeline** integrating multiple feature types
- Combines OpenSMILE features with dimensional labels
- Creates speaker-aware train/val/test splits
- Supports both librosa and OpenSMILE feature extraction
- Outputs training-ready data structures

### **Utilities and Visualization**

#### `utils.py` - Utility Functions
**Comprehensive utility functions** for experiments and analysis
- `plot_training_curves()` - Individual metric visualization
- `plot_loss_curves()` - Combined loss and CCC plotting
- `plot_dropout_analysis()` - Dropout rate comparison plots
- `statistical_significance_test()` - Statistical analysis
- `ExperimentLogger` - Experiment tracking and logging
- `set_seed()` - Reproducibility utilities

#### `visualize_model.py` - Model Architecture Visualization
**Professional model visualization** for documentation
- Network architecture diagrams
- Model comparison charts
- Publication-quality figure generation

## **Configuration System**

### **config.yaml Structure**
```yaml
dataset: # Dataset configuration
feature_dim: 6373
partition: "speaker_independent"

model: # Model architecture
hidden_layers: 2
hidden_size: 256
dropout_rate: 0.5

training: # Training parameters
batch_size: 16
epochs: 100
learning_rate: 0.001
validation_frequency: 5

experiments: # Experiment setup
dropout_rates: [0.0, 0.3, 0.5, 0.7]
attributes: ["valence", "arousal", "dominance"]
```

## **Testing Framework**

### **Unit Tests**
- **Data loading**: Verify dataset integrity and format
- **Model creation**: Test architecture instantiation
- **Loss computation**: Validate CCC calculation
- **Feature extraction**: Test OpenSMILE integration

### **Integration Tests** 
- **Training pipeline**: End-to-end training verification
- **Checkpointing**: Model saving and loading
- **Visualization**: Plot generation and saving
- **Performance**: Benchmark training speed and accuracy

## **Performance Benchmarks**

### **Training Speed (CPU)**
- **20 epochs**: ~5 minutes
- **100 epochs**: ~20-25 minutes
- **Processing rate**: ~8-9 seconds per epoch

### **Memory Usage**
- **Model size**: ~13MB (3.4M parameters)
- **Dataset**: ~50MB (2,500 × 6,373 features)
- **Checkpoints**: ~7MB (includes training history)

### **Expected Performance**
- **Valence CCC**: 0.08-0.12 (challenging emotion)
- **Arousal CCC**: 0.15-0.25 (easier to predict)
- **Dominance CCC**: 0.12-0.20 (moderate difficulty)

## **Development Workflow**

### **Adding New Features**
1. **Update relevant module** (model.py, data_loader.py, etc.)
2. **Add configuration options** in config.yaml
3. **Create test script** in examples/
4. **Update documentation** in module docstrings
5. **Test integration** with existing pipeline

### **Performance Optimization**
1. **Profile bottlenecks** using Python profiling tools
2. **Optimize data loading** with PyTorch DataLoader tuning
3. **Memory optimization** for large feature matrices
4. **GPU acceleration** for faster training

## **Extension Points**

### **Easy Extensions**
- **New datasets**: Add loader functions in data_loader.py
- **New architectures**: Add models in model.py
- **New loss functions**: Add to loss.py
- **New visualizations**: Add to utils.py

### **Advanced Extensions**
- **Attention mechanisms**: Modify model.py for interpretability
- **Multi-task learning**: Extend MultiTaskEmotionMLP
- **Transfer learning**: Add pre-trained model integration
- **Real-time inference**: Add streaming audio processing

## **API Reference**

Each module contains comprehensive docstrings with:
- **Function signatures** with type hints
- **Parameter descriptions** with expected formats
- **Return value specifications** with shapes and types
- **Usage examples** with realistic scenarios
- **Error handling** with common failure modes

For detailed API documentation, see the docstrings in each source file.