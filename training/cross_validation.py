"""
Walk-Forward Cross-Validation Framework

This module provides a walk-forward expanding window cross-validation framework for Bitcoin price prediction.
It implements the specific time-based fold structure for proper temporal validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import os
from pathlib import Path
from tqdm import tqdm
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle

# Import our components
from preprocessing.precog_preprocess import preprocess_bitcoin_enhanced
from predictors import create_predictor_from_config
from losses import precog_loss, synth_loss, evaluate_precog, evaluate_synth

warnings.filterwarnings('ignore')

@dataclass
class WalkForwardResults:
    """Results from walk-forward cross-validation."""
    fold_scores: List[float]
    fold_metrics: List[Dict[str, Any]]
    fold_periods: List[Dict[str, str]]
    mean_score: float
    std_score: float
    best_fold: int
    best_score: float
    config: Dict[str, Any]
    cv_summary: Dict[str, Any]

class WalkForwardCrossValidator:
    """
    Walk-forward expanding window cross-validation for Bitcoin price prediction.
    
    This class implements the specific time-based fold structure:
    - Fold 1: Train 2020–2023 Q3, Val 2023 Q4
    - Fold 2: Train 2020–2024 Q1, Val 2024 Q2
    - Fold 3: Train 2020–2024 Q2, Val 2024 Q3
    - Final: Train 2020–2025 Q1, Val Final holdout
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 save_models: bool = True,
                 verbose: bool = True):
        """
        Initialize the walk-forward cross-validator.
        
        Args:
            config: Configuration dictionary
            save_models: Whether to save models from each fold
            verbose: Whether to print detailed progress
        """
        self.config = config
        self.save_models = save_models
        self.verbose = verbose
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        
        # Create save directory
        self.save_dir = Path(config.get('save_dir', 'models')) / 'walk_forward_cv'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Define walk-forward periods
        self.fold_periods = [
            {
                'name': 'Fold_1',
                'train_start': '2020-01-01',
                'train_end': '2023-09-30',
                'val_start': '2023-10-01', 
                'val_end': '2023-12-31'
            },
            {
                'name': 'Fold_2',
                'train_start': '2020-01-01',
                'train_end': '2024-03-31',
                'val_start': '2024-04-01',
                'val_end': '2024-06-30'
            },
            {
                'name': 'Fold_3',
                'train_start': '2020-01-01', 
                'train_end': '2024-06-30',
                'val_start': '2024-07-01',
                'val_end': '2024-09-30'
            },
            {
                'name': 'Final',
                'train_start': '2020-01-01',
                'train_end': '2025-03-31',
                'val_start': '2025-04-01',
                'val_end': '2025-12-31'
            }
        ]
        
        if self.verbose:
            print(f"WalkForwardCrossValidator initialized:")
            print(f"  Mode: {config['mode']}")
            print(f"  Model: {config['model_type']}")
            print(f"  Folds: {len(self.fold_periods)}")
            print(f"  Device: {self.device}")
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load and preprocess the full dataset with timestamps.
        
        Returns:
            Tuple of (timestamp_df, X_full, y_full)
        """
        print("Loading and preprocessing full dataset...")
        
        # Use the enhanced preprocessing pipeline
        train_data, val_data, test_data, preprocessor = preprocess_bitcoin_enhanced(
            dataset_path=self.config.get('dataset_path', 'datasets/complete_dataset_20250709_152829.csv'),
            lookback=self.config.get('lookback', 12),
            horizon=self.config.get('horizon', 12),
            scaler_type=self.config.get('scaler_type', 'standard'),
            save_dir=None,  # Don't save artifacts for CV
            bias=self.config.get('bias', 15000)
        )
        
        # Combine all data
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        X_full = np.vstack([X_train, X_val, X_test])
        y_full = np.vstack([y_train, y_val, y_test])
        
        # Load raw data to get timestamps
        df_raw = pd.read_csv(self.config.get('dataset_path', 'datasets/complete_dataset_20250709_152829.csv'))
        
        # Handle timestamp column
        if 'timestamp' in df_raw.columns:
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        elif df_raw.index.dtype == 'datetime64[ns]':
            df_raw = df_raw.reset_index()
            df_raw.rename(columns={'index': 'timestamp'}, inplace=True)
        else:
            df_raw = df_raw.reset_index()
            df_raw['timestamp'] = pd.to_datetime(df_raw['index'])
        
        # Apply same bias as preprocessing
        bias = self.config.get('bias', 15000)
        if bias > 0:
            df_raw = df_raw.iloc[bias:].copy()
        
        # Account for sequence creation (skip first lookback samples)
        lookback = self.config.get('lookback', 12)
        df_timestamps = df_raw.iloc[lookback:].copy()
        
        # Ensure lengths match
        min_len = min(len(X_full), len(df_timestamps))
        X_full = X_full[:min_len]
        y_full = y_full[:min_len]
        df_timestamps = df_timestamps.iloc[:min_len].copy()
        
        print(f"Full dataset loaded:")
        print(f"  Samples: {len(X_full)}")
        print(f"  Features: {X_full.shape[1]}")
        print(f"  Targets: {y_full.shape[1]}")
        print(f"  Time range: {df_timestamps['timestamp'].min()} to {df_timestamps['timestamp'].max()}")
        
        # Store preprocessor for later use
        self.preprocessor = preprocessor
        
        return df_timestamps, X_full, y_full
    
    def create_fold_splits(self, df_timestamps: pd.DataFrame, X_full: np.ndarray, y_full: np.ndarray) -> List[Dict[str, Any]]:
        """
        Create walk-forward time-based splits.
        
        Args:
            df_timestamps: DataFrame with timestamps
            X_full: Full feature array
            y_full: Full target array
            
        Returns:
            List of fold data dictionaries
        """
        print("Creating walk-forward time-based splits...")
        
        fold_data = []
        timestamps = df_timestamps['timestamp']
        
        for i, period in enumerate(self.fold_periods):
            print(f"\n  Creating {period['name']}:")
            print(f"    Train: {period['train_start']} to {period['train_end']}")
            print(f"    Val: {period['val_start']} to {period['val_end']}")
            
            # Parse dates
            train_start = pd.to_datetime(period['train_start'])
            train_end = pd.to_datetime(period['train_end'])
            val_start = pd.to_datetime(period['val_start'])
            val_end = pd.to_datetime(period['val_end'])
            
            # Create boolean masks
            train_mask = (timestamps >= train_start) & (timestamps <= train_end)
            val_mask = (timestamps >= val_start) & (timestamps <= val_end)
            
            # Get indices
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            
            # Extract data
            X_train_fold = X_full[train_indices]
            y_train_fold = y_full[train_indices]
            X_val_fold = X_full[val_indices]
            y_val_fold = y_full[val_indices]
            
            # Validate splits
            if len(X_train_fold) == 0:
                print(f"    Warning: No training data for {period['name']}")
                continue
            if len(X_val_fold) == 0:
                print(f"    Warning: No validation data for {period['name']}")
                continue
            
            print(f"    Train samples: {len(X_train_fold)}")
            print(f"    Val samples: {len(X_val_fold)}")
            
            # Fit scaler on this fold's training data
            fold_scaler = type(self.preprocessor.scaler)()
            fold_scaler.fit(X_train_fold)
            
            # Transform features for this fold
            X_train_scaled = fold_scaler.transform(X_train_fold)
            X_val_scaled = fold_scaler.transform(X_val_fold)
            
            fold_data.append({
                'name': period['name'],
                'period': period,
                'X_train': X_train_scaled,
                'y_train': y_train_fold,
                'X_val': X_val_scaled,
                'y_val': y_val_fold,
                'scaler': fold_scaler,
                'train_indices': train_indices,
                'val_indices': val_indices
            })
        
        print(f"\nCreated {len(fold_data)} valid folds")
        return fold_data
    
    def create_data_loaders(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for a fold.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        batch_size = self.config.get('batch_size', 512)
        num_workers = self.config.get('num_workers', 0)  # Set to 0 for compatibility
        
        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Can shuffle within fold since we already have time-based splits
            num_workers=num_workers,
            pin_memory=self.config.get('pin_memory', True) and torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.config.get('pin_memory', True) and torch.cuda.is_available()
        )
        
        return train_loader, val_loader
    
    def train_fold(self, fold_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Train a model for a specific fold.
        
        Args:
            fold_data: Dictionary containing fold data
            
        Returns:
            Tuple of (best_score, metrics_dict)
        """
        fold_name = fold_data['name']
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training {fold_name}")
            print(f"{'='*60}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            fold_data['X_train'], fold_data['y_train'],
            fold_data['X_val'], fold_data['y_val']
        )
        
        # Get input size
        input_size = fold_data['X_train'].shape[1]
        
        # Create model
        model = create_predictor_from_config(self.config, input_size)
        model.to(self.device)
        
        if self.verbose:
            print(f"Model created: {model.__class__.__name__}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.get('lr', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.get('lr', 1e-4),
            steps_per_epoch=len(train_loader),
            epochs=self.config.get('epochs', 10)
        )
        
        # Setup mixed precision
        scaler = None
        if self.config.get('mixed_precision', True) and torch.cuda.is_available():
            try:
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
            except ImportError:
                pass
        
        # Training loop
        best_score = float('inf')
        patience = self.config.get('patience', 5)
        patience_counter = 0
        training_history = {'train_losses': [], 'val_scores': []}
        
        epochs = self.config.get('epochs', 10)
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_batches = 0
            
            if self.verbose:
                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Train)', 
                                leave=False, ncols=100)
            else:
                train_pbar = train_loader
            
            for batch_X, batch_y in train_pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Skip NaN batches
                if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass
                if scaler is not None:
                    try:
                        from torch.cuda.amp import autocast
                        with autocast():
                            predictions = model(batch_X)
                            loss = self._calculate_loss(predictions, batch_y)
                        
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                     self.config.get('grad_clip', 1.0))
                        scaler.step(optimizer)
                        scaler.update()
                    except:
                        # Fallback to non-mixed precision
                        predictions = model(batch_X)
                        loss = self._calculate_loss(predictions, batch_y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                     self.config.get('grad_clip', 1.0))
                        optimizer.step()
                else:
                    predictions = model(batch_X)
                    loss = self._calculate_loss(predictions, batch_y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                 self.config.get('grad_clip', 1.0))
                    optimizer.step()
                
                scheduler.step()
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    train_loss += loss.item()
                    train_batches += 1
                
                if self.verbose and hasattr(train_pbar, 'set_postfix'):
                    train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Validation phase
            val_score = self._evaluate_fold(model, val_loader)
            
            # Update history
            avg_train_loss = train_loss / max(train_batches, 1)
            training_history['train_losses'].append(avg_train_loss)
            training_history['val_scores'].append(val_score)
            
            if self.verbose:
                print(f"  Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={val_score:.4f}")
            
            # Check for best score
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                
                if self.save_models:
                    # Save best model for this fold
                    fold_dir = self.save_dir / fold_name
                    fold_dir.mkdir(exist_ok=True)
                    
                    model_path = fold_dir / 'best_model.pth'
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': self.config,
                        'fold_name': fold_name,
                        'epoch': epoch,
                        'score': best_score,
                        'training_history': training_history,
                        'fold_period': fold_data['period']
                    }, model_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Final comprehensive evaluation
        final_metrics = self._evaluate_fold_comprehensive(model, val_loader, fold_data['y_val'])
        final_metrics.update({
            'fold_name': fold_name,
            'best_score': best_score,
            'epochs_trained': epoch + 1,
            'training_history': training_history,
            'period': fold_data['period']
        })
        
        return best_score, final_metrics
    
    def _calculate_loss(self, predictions: Tuple[torch.Tensor, ...], targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on model mode."""
        if self.config['mode'] == 'precog':
            # For precog mode: predictions should be (point_pred, interval_pred)
            if isinstance(predictions, tuple) and len(predictions) >= 2:
                point_pred, interval_pred = predictions[0], predictions[1]
            else:
                # If not tuple, split the predictions
                point_pred = predictions[..., :targets.shape[1]//3]  # First third
                interval_pred = predictions[..., targets.shape[1]//3:]  # Rest for intervals
            return precog_loss(point_pred, interval_pred, targets, None)  # No scaler for relative returns
        elif self.config['mode'] == 'synth':
            # For synth mode: detailed predictions
            detailed_pred = predictions[0] if isinstance(predictions, tuple) else predictions
            return synth_loss(detailed_pred, targets, None)  # No scaler for relative returns
        else:
            # Default MSE loss for relative returns
            return nn.MSELoss()(predictions, targets)
    
    def _evaluate_fold(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Quick evaluation for a fold (returns single score)."""
        model.eval()
        total_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                    continue
                
                predictions = model(batch_X)
                loss = self._calculate_loss(predictions, batch_y)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    batch_count += 1
        
        return total_loss / max(batch_count, 1)
    
    def _evaluate_fold_comprehensive(self, model: nn.Module, val_loader: DataLoader, y_val: np.ndarray) -> Dict[str, Any]:
        """Comprehensive evaluation for a fold."""
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                    continue
                
                predictions = model(batch_X)
                
                # Handle different prediction formats
                if isinstance(predictions, tuple):
                    pred_tensor = predictions[0]  # Take first element
                else:
                    pred_tensor = predictions
                
                all_predictions.append(pred_tensor.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        if len(all_predictions) == 0:
            return {'mse': float('inf'), 'mae': float('inf')}
        
        # Concatenate all predictions and targets
        pred_array = np.vstack(all_predictions)
        target_array = np.vstack(all_targets)
        
        # Calculate metrics
        mse = np.mean((pred_array - target_array) ** 2)
        mae = np.mean(np.abs(pred_array - target_array))
        
        # Calculate per-timestep metrics for relative returns
        timestep_metrics = {}
        horizon = self.config.get('horizon', 12)
        
        # Assuming targets are structured as [point, min, max] for each timestep
        for i in range(horizon):
            if i * 3 + 2 < target_array.shape[1]:  # Ensure we have enough columns
                point_mse = np.mean((pred_array[:, i*3] - target_array[:, i*3]) ** 2)
                timestep_metrics[f'timestep_{i}_point_mse'] = point_mse
        
        return {
            'mse': mse,
            'mae': mae,
            'timestep_metrics': timestep_metrics,
            'n_samples': len(target_array)
        }
    
    def run_walk_forward_cv(self) -> WalkForwardResults:
        """
        Run complete walk-forward cross-validation.
        
        Returns:
            WalkForwardResults object with all results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Starting Walk-Forward Cross-Validation")
            print(f"{'='*80}")
        
        # Load and preprocess data
        df_timestamps, X_full, y_full = self.load_and_preprocess_data()
        
        # Create fold splits
        fold_data_list = self.create_fold_splits(df_timestamps, X_full, y_full)
        
        # Run cross-validation
        fold_scores = []
        fold_metrics = []
        fold_periods = []
        
        for fold_data in fold_data_list:
            # Train fold
            score, metrics = self.train_fold(fold_data)
            
            fold_scores.append(score)
            fold_metrics.append(metrics)
            fold_periods.append(fold_data['period'])
            
            if self.verbose:
                print(f"{fold_data['name']} completed: Score = {score:.4f}")
        
        # Calculate summary statistics
        if len(fold_scores) > 0:
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            best_fold = np.argmin(fold_scores)
            best_score = fold_scores[best_fold]
        else:
            mean_score = float('inf')
            std_score = 0
            best_fold = -1
            best_score = float('inf')
        
        # Create comprehensive summary
        cv_summary = {
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'best_fold': int(best_fold),
            'best_score': float(best_score),
            'fold_scores': [float(s) for s in fold_scores],
            'mode': self.config['mode'],
            'model_type': self.config['model_type'],
            'n_folds': len(fold_data_list),
            'total_samples': len(X_full),
            'timestamp': datetime.now().isoformat(),
            'cv_type': 'walk_forward_expanding_window'
        }
        
        # Save results
        results_path = self.save_dir / 'cv_results.json'
        with open(results_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Walk-Forward Cross-Validation Results")
            print(f"{'='*80}")
            print(f"Mean Score: {mean_score:.4f} ± {std_score:.4f}")
            if best_fold >= 0:
                print(f"Best Fold: {fold_periods[best_fold]['name']} (Score: {best_score:.4f})")
            print(f"Individual Scores: {[f'{s:.4f}' for s in fold_scores]}")
            print(f"Results saved to: {results_path}")
        
        return WalkForwardResults(
            fold_scores=fold_scores,
            fold_metrics=fold_metrics,
            fold_periods=fold_periods,
            mean_score=mean_score,
            std_score=std_score,
            best_fold=best_fold,
            best_score=best_score,
            config=self.config,
            cv_summary=cv_summary
        )


# Convenience function
def run_walk_forward_cv(config: Dict[str, Any], 
                       save_models: bool = True, 
                       verbose: bool = True) -> WalkForwardResults:
    """
    Convenience function to run walk-forward cross-validation.
    
    Args:
        config: Configuration dictionary
        save_models: Whether to save models
        verbose: Whether to print progress
        
    Returns:
        WalkForwardResults object
    """
    cv = WalkForwardCrossValidator(config, save_models, verbose)
    return cv.run_walk_forward_cv()


# Example configuration for testing
def get_example_config():
    """Get example configuration for walk-forward CV."""
    return {
        'mode': 'precog',
        'model_type': 'enhanced_transformer',
        'dataset_path': 'datasets/complete_dataset_20250709_152829.csv',
        'lookback': 12,
        'horizon': 12,
        'scaler_type': 'standard',
        'bias': 15000,
        'batch_size': 512,
        'epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'patience': 5,
        'grad_clip': 1.0,
        'mixed_precision': True,
        'use_gpu': True,
        'save_dir': 'models/walk_forward',
        # Model-specific parameters
        'hidden_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.1
    }


if __name__ == "__main__":
    # Test walk-forward cross-validation
    config = get_example_config()
    
    print("Testing Walk-Forward Cross-Validation...")
    results = run_walk_forward_cv(config, save_models=True, verbose=True)
    
    print(f"\nFinal Results:")
    print(f"Mean Score: {results.mean_score:.4f} ± {results.std_score:.4f}")
    print(f"Best Fold: {results.best_fold}")
    print(f"Best Score: {results.best_score:.4f}")
    print(f"Fold Scores: {results.fold_scores}") 