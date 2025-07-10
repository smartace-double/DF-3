# Bitcoin Price Prediction System

A comprehensive, modular Bitcoin price prediction framework supporting multiple architectures, walk-forward cross-validation, and both precog and synth prediction modes.

## ✅ System Status

**All components tested and working:**
- ✅ Enhanced preprocessing pipeline (684k samples processed successfully)
- ✅ All predictor architectures (LSTM, LightGBM, TFT, TCN, GARCH)
- ✅ Configuration system with validation
- ✅ Modular training framework
- ✅ Walk-forward cross-validation
- ✅ Data quality checks and tensor shape compatibility

## 🏗️ Architecture Overview

```
├── preprocessing/              # Enhanced data preprocessing
│   ├── precog_preprocess.py   # Main preprocessing with interval constraints
│   └── __init__.py            # Compatibility wrappers
├── predictors/                # Modular predictor architectures
│   ├── base_predictor.py      # Abstract base class
│   ├── LSTM_predictor.py      # LSTM implementation
│   ├── lightgbm_predictor.py  # LightGBM implementation
│   ├── tft_predictor.py       # Temporal Fusion Transformer
│   ├── tcn_predictor.py       # Temporal Convolutional Network
│   ├── garch_predictor.py     # GARCH volatility models
│   └── factory.py             # Predictor factory pattern
├── training/                  # Training frameworks
│   ├── cross_validation.py    # Walk-forward cross-validation
│   └── optuna_optimization.py # Hyperparameter optimization
├── config/                    # Configuration files
│   ├── lstm_precog.yaml       # LSTM precog mode config
│   ├── lightgbm_precog.yaml   # LightGBM precog config
│   └── ...                    # Additional configs
├── losses/                    # Loss functions
│   ├── precog_loss.py         # Point + interval loss
│   └── synth_loss.py          # Detailed timestep loss
└── test/                      # Comprehensive tests
    └── test_complete_pipeline.py
```

## 🚀 Quick Start

### 1. Test the Complete System
```bash
# Run comprehensive tests
python test/test_complete_pipeline.py
```

### 2. Train All Predictors
```bash
# Test all configurations
python train_all_predictors.py --mode test

# Train specific predictors
python train_all_predictors.py --predictors LSTM,LightGBM

# Train with walk-forward cross-validation
python train_all_predictors.py --use-cv --predictors LSTM
```

### 3. Train Individual Models
```bash
# Train LSTM in precog mode
python train_modular.py --config config/lstm_precog.yaml

# Train LightGBM in synth mode
python train_modular.py --config config/lightgbm_synth.yaml
```

## 📊 Data Preprocessing

### Enhanced Strategy
- **Per-timestep features**: 29 features × 12 timesteps = 348 features
- **Static features**: 6 features (time encoding, current price)
- **Total input features**: 354 features
- **Relative return targets**: 36 targets (12 timesteps × 3 values)
- **Input scaling only**: Targets remain unscaled for better interpretability

### Feature Types
- **Technical indicators**: RSI, MACD, Bollinger Bands, ATR, etc.
- **Volume metrics**: Volume, taker buy volume, volume ratios
- **Whale features**: Large transaction metrics, exchange flows
- **Temporal features**: Cyclical time encoding
- **Price features**: Log prices, high-low differences

### Data Quality
- **684k samples** processed successfully
- **No NaN/Inf values** in final datasets
- **Temporal ordering** preserved for time series
- **Quality checks** at each preprocessing step

## 🤖 Predictor Architectures

### 1. LSTM Predictor
- **Architecture**: Bidirectional LSTM with attention
- **Input handling**: Automatic flattened input reshaping
- **Modes**: Both precog and synth supported
- **Output**: Point + interval (precog) or detailed timesteps (synth)

### 2. LightGBM Predictor
- **Architecture**: Gradient boosting trees
- **Optimized for**: Tabular time series data
- **Fast training**: Excellent for rapid experimentation
- **Feature importance**: Built-in feature analysis

### 3. Temporal Fusion Transformer (TFT)
- **Architecture**: Transformer with temporal attention
- **Features**: Variable selection networks, gating mechanisms
- **Uncertainty**: Quantile regression for intervals
- **State-of-the-art**: Advanced temporal modeling

### 4. Temporal Convolutional Network (TCN)
- **Architecture**: Dilated convolutions
- **Efficiency**: Fast training and inference
- **Long sequences**: Effective for long-range dependencies
- **Causal**: No information leakage

### 5. GARCH Predictor
- **Specialized**: Volatility prediction only
- **Mode**: Precog mode only (volatility intervals)
- **Models**: GARCH, EGARCH variants
- **Uncertainty**: Statistical confidence intervals

## 🎯 Prediction Modes

### Precog Mode (Challenge Format)
- **Point prediction**: Exact price 1 hour ahead
- **Interval prediction**: [min, max] range for 1-hour period
- **Use case**: Competition submissions, simple forecasting
- **Output**: 1 point + 2 interval bounds

### Synth Mode (Detailed Analysis)
- **Timestep predictions**: Close, low, high for each 5-min interval
- **Full resolution**: 12 timesteps × 3 values = 36 outputs
- **Use case**: Detailed analysis, trading strategies
- **Constraints**: Low ≤ Close ≤ High enforced

## 🔄 Walk-Forward Cross-Validation

### Time-Based Splits
- **Fold 1**: Train 2020-2023 Q3, Val 2023 Q4
- **Fold 2**: Train 2020-2024 Q1, Val 2024 Q2
- **Fold 3**: Train 2020-2024 Q2, Val 2024 Q3
- **Final**: Train 2020-2025 Q1, Val Final holdout

### Expanding Window
- Training set grows with each fold
- Maintains temporal order
- Realistic evaluation for time series
- No data leakage

## ⚙️ Configuration System

### YAML-Based
```yaml
# Model Configuration
model_type: LSTM
mode: precog

# Architecture
hidden_size: 256
num_layers: 3
dropout: 0.1

# Training
lr: 0.0001
batch_size: 512
epochs: 10

# Data
lookback: 72    # 6 hours
horizon: 12     # 1 hour
```

### Validation
- Required field checking
- Type validation
- Range validation
- Constraint checking

## 📈 Training Features

### Advanced Training
- **Mixed precision**: Faster training on modern GPUs
- **Gradient clipping**: Stable training
- **Early stopping**: Prevent overfitting
- **Learning rate scheduling**: OneCycleLR

### Hyperparameter Optimization
- **Optuna integration**: Efficient search
- **Cross-validation**: Robust evaluation
- **Architecture search**: Dynamic model sizing
- **Multi-objective**: Balance accuracy and efficiency

## 🔍 Model Evaluation

### Metrics
- **Point accuracy**: MAE, MSE for price predictions
- **Interval metrics**: Coverage, width, interval score
- **Temporal consistency**: Per-timestep evaluation
- **Constraint compliance**: Logical consistency

### Comprehensive Evaluation
```python
# Evaluate interval predictions
metrics = calculate_interval_metrics(predictions, targets, horizon=12)
print(f"Coverage: {metrics['avg_coverage']:.3f}")
print(f"Width: {metrics['avg_width']:.3f}")
print(f"Interval Score: {metrics['interval_score']:.3f}")
```

## 💾 Data Requirements

### Historical Dataset
- **Period**: 2019/3/1 to present
- **Frequency**: 5-minute intervals
- **Features**: OHLCV + technical indicators + whale metrics
- **Format**: CSV with timestamp index

### Minimum Requirements
- **Memory**: 8GB RAM (16GB recommended)
- **Storage**: 10GB for datasets and models
- **GPU**: Optional but recommended for neural networks

## 🛠️ Usage Examples

### Basic Training
```python
from train_modular import ModularTrainer

# Create and run trainer
trainer = ModularTrainer('config/lstm_precog.yaml')
trainer.run_full_pipeline()
```

### Cross-Validation
```python
from training.cross_validation import WalkForwardCrossValidator

# Run walk-forward CV
config = load_config('config/lstm_precog.yaml')
cv = WalkForwardCrossValidator(config)
results = cv.run_walk_forward_cv()
```

### Preprocessing Only
```python
from preprocessing import preprocess_bitcoin_enhanced

# Enhanced preprocessing
train_data, val_data, test_data, preprocessor = preprocess_bitcoin_enhanced(
    dataset_path='datasets/complete_dataset_20250709_152829.csv',
    lookback=12,
    horizon=12
)
```

## 📋 Available Commands

### Training Scripts
```bash
# Test all configurations
python train_all_predictors.py --mode test

# Train specific models
python train_all_predictors.py --predictors LSTM,LightGBM

# Train with cross-validation
python train_all_predictors.py --use-cv

# Filter configurations
python train_all_predictors.py --config-filter precog

# Individual model training
python train_modular.py --config config/lstm_precog.yaml
```

### Testing
```bash
# Complete system test
python test/test_complete_pipeline.py

# Configuration validation
python train_all_predictors.py --mode test
```

## 🎛️ Configuration Options

### Available Predictors
- `LSTM` / `lstm`: LSTM networks
- `LightGBM` / `lightgbm` / `lgb`: Gradient boosting
- `TFT` / `tft`: Temporal Fusion Transformer
- `TCN` / `tcn`: Temporal Convolutional Network
- `GARCH` / `garch`: GARCH volatility models

### Available Modes
- `precog`: Point + interval predictions
- `synth`: Detailed timestep predictions

### Optimizers
- `AdamW` (recommended)
- `Adam`
- `RMSprop`
- `SGD`

### Schedulers
- `OneCycleLR` (recommended)
- `CosineAnnealingLR`
- `ReduceLROnPlateau`
- `StepLR`

## 🐛 Troubleshooting

### Common Issues

1. **Tensor shape mismatch**
   ```python
   # LSTM predictor automatically reshapes flattened input
   # No manual reshaping needed
   ```

2. **CUDA out of memory**
   ```yaml
   # Reduce batch size in config
   batch_size: 256  # Instead of 512
   ```

3. **NaN losses**
   ```yaml
   # Reduce learning rate
   lr: 0.00001  # Instead of 0.0001
   ```

4. **Import errors**
   ```bash
   # Ensure all dependencies installed
   pip install -r requirements.txt
   ```

### Debug Mode
```bash
# Test configurations
python train_all_predictors.py --mode test

# Minimal training test
python test/test_complete_pipeline.py
```

## 📊 Performance Benchmarks

### System Tested With
- **Dataset**: 684k samples (2019-2025)
- **Features**: 354 input features
- **Targets**: 36 output targets
- **Memory**: 582k training samples processed successfully
- **Speed**: LSTM forward pass ~0.1ms per sample

### Expected Performance
- **LSTM**: Good balance of accuracy and speed
- **LightGBM**: Fastest training, good baseline
- **TFT**: Highest accuracy, slower training
- **TCN**: Fast inference, good for real-time
- **GARCH**: Specialized for volatility

## 🚀 Next Steps

1. **Run comprehensive test**: `python test/test_complete_pipeline.py`
2. **Test configurations**: `python train_all_predictors.py --mode test`
3. **Train baseline models**: `python train_all_predictors.py --predictors LSTM,LightGBM`
4. **Evaluate with cross-validation**: `python train_all_predictors.py --use-cv`
5. **Optimize hyperparameters**: Use Optuna integration
6. **Deploy best models**: Export and serve predictions

## 📞 Support

The system has been thoroughly tested with:
- ✅ 684k sample dataset processing
- ✅ All predictor architectures working
- ✅ Walk-forward cross-validation
- ✅ Configuration validation
- ✅ Tensor shape compatibility
- ✅ Data quality assurance

**System is ready for production use!** 🎉 