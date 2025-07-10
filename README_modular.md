# Modular Bitcoin Price Prediction Framework

This framework provides a sophisticated, modular approach to training Bitcoin price prediction models with configurable architectures and training settings.

## Features

- **Modular Architecture**: Separate predictor classes that can be easily extended
- **Configuration-Based**: YAML configuration files for easy experimentation
- **Multiple Modes**: Support for both challenge mode (point + interval) and detailed mode (full timestep predictions)
- **Pluggable Models**: Easy to add new model architectures
- **Comprehensive Training**: Advanced training features including mixed precision, early stopping, and learning rate scheduling

## Directory Structure

```
.
├── config/                          # Configuration files
│   ├── config_loader.py            # Configuration loading utilities
│   ├── lstm_challenge.yaml         # LSTM challenge mode config
│   ├── lstm_detailed.yaml          # LSTM detailed mode config
│   └── lstm_challenge_optimized.yaml # Optimized LSTM config
├── predictors/                      # Model architectures
│   ├── __init__.py                 # Package initialization
│   ├── base_predictor.py           # Base class for all predictors
│   ├── LSTM_predictor.py           # LSTM implementation
│   └── factory.py                  # Model factory
├── train_modular.py                # New modular training script
├── train_deepseek_1.py             # Original training script
└── README_modular.md               # This file
```

## Quick Start

### 1. Training with Default Configuration

```bash
# Train LSTM model in challenge mode
python train_modular.py --config config/lstm_challenge.yaml

# Train LSTM model in detailed mode
python train_modular.py --config config/lstm_detailed.yaml

# Train with optimized settings
python train_modular.py --config config/lstm_challenge_optimized.yaml
```

### 2. Testing Configuration

```bash
# Test configuration loading
python train_modular.py --config config/lstm_challenge.yaml --mode test
```

### 3. Evaluating Existing Model

```bash
# Evaluate a trained model
python train_modular.py --config config/lstm_challenge.yaml --mode evaluate --model_path models/best_model.pth
```

## Configuration Files

### Basic Configuration Structure

```yaml
# Model Configuration
model_type: LSTM          # Model architecture
mode: challenge           # 'challenge' or 'detailed'

# Model Architecture
hidden_size: 256          # Hidden dimension size
num_layers: 3             # Number of LSTM layers
dropout: 0.1              # Dropout rate
activation: SiLU          # Activation function
use_layer_norm: true      # Use layer normalization
bidirectional: false      # Use bidirectional LSTM

# Training Configuration
lr: 0.0001                # Learning rate
batch_size: 512           # Batch size
epochs: 10                # Number of training epochs
weight_decay: 0.00001     # Weight decay
grad_clip: 1.0            # Gradient clipping
patience: 5               # Early stopping patience

# Data Configuration
dataset_path: datasets/complete_dataset_20250709_152829.csv
lookback: 72              # Input sequence length (6 hours)
horizon: 12               # Output sequence length (1 hour)
train_split: 0.85         # Training data split
val_split: 0.10           # Validation data split
test_split: 0.05          # Test data split

# System Configuration
save_dir: models          # Model save directory
use_gpu: true             # Use GPU if available
mixed_precision: true     # Use mixed precision training
num_workers: 4            # Data loader workers
```

### Available Configurations

1. **`lstm_challenge.yaml`**: Basic LSTM configuration for challenge mode
2. **`lstm_detailed.yaml`**: LSTM configuration for detailed predictions
3. **`lstm_challenge_optimized.yaml`**: Optimized LSTM for maximum performance

## Prediction Modes

### Challenge Mode (`mode: challenge`)
- Predicts **point estimate** (exact price 1 hour ahead)
- Predicts **interval estimate** (min/max price range for 1-hour period)
- Optimized for precog subnet scoring system
- Uses `challenge_loss` function

### Detailed Mode (`mode: detailed`)  
- Predicts **close, low, high** prices for each 5-minute interval
- Provides full temporal resolution (12 timesteps × 3 values = 36 outputs)
- Useful for detailed analysis and research
- Uses `detailed_loss` function (MSE)

## Model Architectures

### LSTM Predictor (`predictors/LSTM_predictor.py`)
- **Architecture**: LSTM encoder with prediction heads
- **Features**: Bidirectional support, layer normalization, multiple activations
- **Modes**: Both challenge and detailed modes supported
- **Outputs**: 
  - Challenge mode: `(point_pred, interval_pred)`
  - Detailed mode: `(detailed_pred,)` with shape `[batch_size, 12, 3]`

### Adding New Predictors

1. Create new predictor class inheriting from `BaseBitcoinPredictor`
2. Implement required methods: `_build_model()` and `forward()`
3. Register in `predictors/factory.py`
4. Add to `predictors/__init__.py`

Example:
```python
# predictors/transformer_predictor.py
from .base_predictor import BaseBitcoinPredictor

class TransformerBitcoinPredictor(BaseBitcoinPredictor):
    def _build_model(self):
        # Implement transformer architecture
        pass
    
    def forward(self, x):
        # Implement forward pass
        pass

# Register in factory.py
PredictorFactory.register_predictor('Transformer', TransformerBitcoinPredictor)
```

## Training Features

### Mixed Precision Training
- Automatic mixed precision with `torch.cuda.amp`
- Faster training on modern GPUs
- Reduced memory usage
- Configurable via `mixed_precision: true/false`

### Early Stopping
- Monitors validation loss
- Configurable patience
- Saves best model automatically

### Learning Rate Scheduling
- OneCycleLR scheduler
- Optimized learning rate progression
- Synchronized with training epochs

### Model Saving
- Automatic saving of best model
- Comprehensive model information
- Training history tracking
- Easy model loading and resuming

## Advanced Usage

### Custom Configuration
```python
from config.config_loader import ConfigLoader

# Create custom configuration
config = {
    'model_type': 'LSTM',
    'mode': 'challenge',
    'hidden_size': 512,
    'num_layers': 4,
    # ... other parameters
}

# Save configuration
loader = ConfigLoader()
loader.save_config(config, 'config/my_config.yaml')
```

### Programmatic Training
```python
from train_modular import ModularTrainer

# Create trainer
trainer = ModularTrainer('config/lstm_challenge.yaml')

# Run training pipeline
success = trainer.run_full_pipeline()
```

### Model Loading and Inference
```python
from predictors import load_predictor

# Load trained model
model = load_predictor('models/best_model.pth')

# Make predictions
predictions = model(input_tensor)
```

## Configuration Validation

The framework includes comprehensive configuration validation:
- **Required keys**: `model_type`, `mode`
- **Type checking**: Ensures correct data types
- **Range validation**: Validates parameter ranges
- **Constraint checking**: Ensures splits sum to 1.0
- **Value validation**: Checks against valid options

## Performance Optimization

### GPU Optimization
- Automatic GPU detection and usage
- Mixed precision training
- Efficient data loading with multiple workers
- Memory pinning for faster GPU transfers

### Memory Management
- Persistent data loader workers
- Gradient accumulation support
- Automatic memory cleanup
- Efficient batch processing

## Error Handling

- **NaN Detection**: Automatic detection and handling of NaN values
- **Graceful Degradation**: Continues training even with some failed batches
- **Comprehensive Logging**: Detailed error messages and debugging information
- **Validation**: Pre-flight validation of configurations and data

## Migration from Original Script

The modular framework is designed to be backward compatible:

1. **Existing Components**: Reuses `BTCDataset`, `challenge_loss`, `detailed_loss` from original script
2. **Same Data Format**: Compatible with existing dataset format
3. **Same Loss Functions**: Uses identical loss calculations
4. **Easy Migration**: Minimal changes needed to existing workflows

## Example Workflows

### Basic Training
```bash
# Train basic LSTM model
python train_modular.py --config config/lstm_challenge.yaml
```

### Hyperparameter Tuning
```bash
# Create multiple configurations with different parameters
# Train each configuration
python train_modular.py --config config/lstm_small.yaml
python train_modular.py --config config/lstm_medium.yaml  
python train_modular.py --config config/lstm_large.yaml
```

### Model Comparison
```bash
# Train challenge mode
python train_modular.py --config config/lstm_challenge.yaml

# Train detailed mode
python train_modular.py --config config/lstm_detailed.yaml

# Compare results
python train_modular.py --config config/lstm_challenge.yaml --mode evaluate --model_path models/best_model.pth
```

## Troubleshooting

### Common Issues

1. **Configuration not found**: Check file path and ensure YAML syntax is correct
2. **CUDA out of memory**: Reduce batch size or model size
3. **NaN losses**: Check data quality and learning rate
4. **Slow training**: Enable mixed precision and increase batch size

### Debug Mode
```bash
# Test configuration loading
python train_modular.py --config config/lstm_challenge.yaml --mode test
```

## Future Extensions

The modular framework is designed for easy extension:

- **New Architectures**: Add Transformers, CNNs, etc.
- **New Loss Functions**: Implement custom loss functions
- **New Modes**: Add specialized prediction modes
- **Advanced Training**: Add techniques like adversarial training
- **Distributed Training**: Support for multi-GPU training 