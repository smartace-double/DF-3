"""
Modular Bitcoin Price Prediction Training Script

This script provides a sophisticated training framework for Bitcoin price prediction
using configurable models and training settings. It supports both precog mode
(point + interval predictions) and synth mode (full timestep predictions).

Key Features:
- Configuration-based model setup
- Pluggable predictor architectures
- Multiple training modes
- Comprehensive evaluation metrics
- Hyperparameter optimization support
- Model saving and loading
- Memory-efficient data loading
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our modular components
from config.config_loader import ConfigLoader, load_config
from predictors import PredictorFactory, create_predictor_from_config

# Import preprocessing module
from preprocessing import BitcoinPreprocessor, preprocess_bitcoin_data

# Import loss functions
from losses import precog_loss, synth_loss

class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class MemoryEfficientDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that loads data on-demand."""
    
    def __init__(self, X, y, chunk_size=10000):
        self.X = X
        self.y = y
        self.chunk_size = chunk_size
        self.n_samples = len(X)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])

class BTCDataset:
    """Dataset class for Bitcoin price prediction using the preprocessing module."""
    
    def __init__(self, dataset_path='datasets/complete_dataset_20250709_152829.csv', lookback=72, horizon=12):
        self.dataset_path = dataset_path
        self.lookback = lookback
        self.horizon = horizon
        self._preprocessor = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_dataset(self):
        """Load and preprocess the dataset using the preprocessing module."""
        print(f"Loading dataset from {self.dataset_path}")
        
        # Use the preprocessing module
        self.train_data, self.val_data, self.test_data, self._preprocessor = preprocess_bitcoin_data(
            dataset_path=self.dataset_path,
            lookback=self.lookback,
            horizon=self.horizon,
            scaler_type='standard',
            save_dir='preprocessing/artifacts'
        )
        
        # Create DataLoaders with memory-efficient settings
        print("Creating memory-efficient DataLoaders...")
        batch_size = min(512, 256)  # Reduce default batch size for memory efficiency
        train_loader = self._create_dataloader(self.train_data, batch_size=batch_size, shuffle=False)
        val_loader = self._create_dataloader(self.val_data, batch_size=batch_size, shuffle=False)
        test_loader = self._create_dataloader(self.test_data, batch_size=batch_size, shuffle=False)
        
        print(f"Memory-efficient DataLoaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def _create_dataloader(self, data_tuple, batch_size=64, shuffle=False):
        """Create a memory-efficient DataLoader from numpy arrays."""
        # Unpack data tuple
        X, y = data_tuple
        
        # Create memory-efficient dataset
        dataset = MemoryEfficientDataset(X, y)
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=min(4, 2),  # Reduce workers for memory efficiency
            pin_memory=True, 
            persistent_workers=False  # Disable persistent workers to save memory
        )
    
    @property
    def scaler(self):
        """Get the scaler from the preprocessor."""
        return self.preprocessor.scaler if self.preprocessor else None
    
    @property
    def preprocessor(self):
        """Get the preprocessor instance."""
        return getattr(self, '_preprocessor', None)

class ModularTrainer:
    """
    Modular trainer for Bitcoin price prediction models.
    
    This trainer provides a flexible framework for training different
    predictor architectures with configurable settings.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the modular trainer.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = load_config(config_path)
        self.config_loader = ConfigLoader()
        
        # Print configuration
        self.config_loader.print_config(self.config)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config['use_gpu'] else 'cpu')
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available() and self.config['use_gpu']:
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Initialize components (will be set up later)
        self.model: Optional[nn.Module] = None  # type: ignore
        self.dataset: Optional[BTCDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        self.scaler: Optional[GradScaler] = None
        
        # Training state
        self.current_epoch = 0
        self.best_score = float('inf')
        self.training_history = {'train_losses': [], 'val_scores': []}
        
    def setup_data(self):
        """Setup data loaders based on configuration."""
        print("Setting up memory-efficient data loaders...")
        
        # Create dataset
        self.dataset = BTCDataset(
            dataset_path=self.config['dataset_path'],
            lookback=self.config['lookback'],
            horizon=self.config['horizon']
        )
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = self.dataset.load_dataset()
        
        print(f"Data setup complete:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
        
    def setup_model(self):
        """Setup model based on configuration."""
        print("Setting up model...")
        
        # Ensure data is loaded first
        if self.train_loader is None:
            self.setup_data()
        
        # Get input size from first batch
        if self.train_loader is not None:
            for batch_X, batch_y in self.train_loader:
                # The preprocessor flattens the features into a single vector
                # Each sample has (lookback * n_per_timestep_features + n_static_features) features
                input_size = batch_X.shape[1]  # Use shape[1] since data is already flattened
                break
        else:
            raise RuntimeError("Train loader not initialized. Call setup_data() first.")
        
        # Create model using factory
        self.model = create_predictor_from_config(self.config, input_size)
        self.model.to(self.device)
        
        # Print model summary
        self.model.print_model_summary()
        
    def setup_training(self):
        """Setup training components (optimizer, scheduler, etc.)."""
        print("Setting up training components...")
        
        # Ensure model is set up first
        if self.model is None:
            self.setup_model()
        
        # Ensure data is loaded
        if self.train_loader is None:
            self.setup_data()
        
        # Check if this is a LightGBM model (or other non-neural network model)
        if self.model is not None and hasattr(self.model, 'parameters'):
            param_list = list(self.model.parameters())
            if len(param_list) == 0:
                # This is a non-neural network model (like LightGBM)
                print("Detected non-neural network model. Skipping optimizer setup.")
                self.optimizer = None
                self.scheduler = None
                self.scaler = None
            else:
                # This is a neural network model
                # Optimizer
                self.optimizer = optim.AdamW(
                    param_list,
                    lr=self.config['lr'],
                    weight_decay=self.config['weight_decay']
                )
            
            # Scheduler
            if self.train_loader is not None and self.optimizer is not None:
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.config['lr'],
                    steps_per_epoch=len(self.train_loader),
                    epochs=self.config['epochs']
                )
            else:
                self.scheduler = None
            
            # Mixed precision scaler
            if self.config['mixed_precision'] and torch.cuda.is_available():
                self.scaler = GradScaler()
                print("Mixed precision training enabled")
        
        print("Training setup complete")
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        if self.model is None or self.optimizer is None or self.train_loader is None:
            raise RuntimeError("Model, optimizer, or data not initialized. Call setup_training() first.")
        
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        # Progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f'Epoch {self.current_epoch+1}/{self.config["epochs"]} (Train)',
            leave=False,
            ncols=100
        )
        
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Skip batch if contains NaN
            if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                continue
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(batch_X)
                    loss = self._calculate_loss(predictions, batch_y)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(batch_X)
                loss = self._calculate_loss(predictions, batch_y, batch_X)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Skip if loss is NaN or inf
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/batch_count:.4f}'
            })
        
        return total_loss / max(batch_count, 1)
    
    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        if self.model is None or self.val_loader is None:
            raise RuntimeError("Model or validation data not initialized. Call setup_training() first.")
        
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f'Epoch {self.current_epoch+1}/{self.config["epochs"]} (Val)',
            leave=False,
            ncols=100
        )
        
        with torch.no_grad():
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Skip batch if contains NaN
                if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                    continue
                
                predictions = self.model(batch_X)
                loss = self._calculate_loss(predictions, batch_y, batch_X)
                
                # Skip if loss is NaN or inf
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return total_loss / max(batch_count, 1)
    
    def _calculate_loss(self, predictions: Tuple[torch.Tensor, ...], targets: torch.Tensor, batch_X: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate loss based on model mode."""
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup_data() first.")
        
        if self.config['mode'] == 'precog':
            if isinstance(predictions, tuple) and len(predictions) >= 2:
                point_pred = predictions[0]
                interval_pred = predictions[1]
            elif torch.is_tensor(predictions):
                point_pred = predictions
                interval_pred = predictions
            else:
                raise ValueError("Predictions format not recognized for precog mode.")
            
            # Extract current prices from targets (37th feature) - no need for batch_X anymore
            return precog_loss(point_pred, interval_pred, targets, self.dataset.scaler, current_prices=None)
        elif self.config['mode'] == 'synth':
            detailed_pred = predictions[0] if isinstance(predictions, tuple) else predictions
            return synth_loss(detailed_pred, targets, self.dataset.scaler)
        else:
            raise ValueError(f"Unknown mode: {self.config['mode']}")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        
        # If this is a non-neural network model (like LightGBM), use .fit() directly
        if self.model is not None and hasattr(self.model, 'parameters') and len(list(self.model.parameters())) == 0:
            print("Detected non-neural network model. Using .fit() method for training.")
            
            # Ensure dataset is loaded
            if self.dataset is None:
                raise RuntimeError("Dataset not initialized. Call setup_data() first.")
            
            # Extract features and targets from stored numpy arrays
            if self.dataset.train_data is None or self.dataset.val_data is None:
                raise RuntimeError("Training or validation data not available.")
            
            # Data is already in the correct format from preprocessing
            X_train, y_train = self.dataset.train_data
            X_val, y_val = self.dataset.val_data
            
            print(f"Training LightGBM with:")
            print(f"  X_train shape: {X_train.shape}")
            print(f"  y_train shape: {y_train.shape}")
            print(f"  X_val shape: {X_val.shape}")
            print(f"  y_val shape: {y_val.shape}")
            
            # Call the model's fit method
            fit_method = getattr(self.model, 'fit', None)
            if fit_method is not None and callable(fit_method):
                fit_method(X_train, y_train, X_val, y_val)
            else:
                raise RuntimeError("Model does not have a fit method.")
            
            self.best_score = 0  # Not meaningful for LightGBM, but set to 0
            self.training_history['train_losses'].append(0)
            self.training_history['val_scores'].append(0)
            
            # Save best model
            self.save_model(is_best=True)
            print(f"  Model trained and saved (non-neural network mode)")
            return
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=self.config['patience'])
        
        # Training loop
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Validate epoch
            val_loss = self.validate_epoch()
            
            # Update history
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_scores'].append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                  f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.save_model(is_best=True)
                print(f"  New best validation score: {self.best_score:.4f}")
            
            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        self.save_model(is_best=False)
        
        print(f"\nTraining completed!")
        print(f"Best validation score: {self.best_score:.4f}")
        
    def save_model(self, is_best: bool = False):
        """Save model and training state."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(exist_ok=True)
        
        # Save paths
        if is_best:
            model_path = save_dir / 'best_model.pth'
            history_path = save_dir / 'best_training_history.json'
        else:
            model_path = save_dir / f'model_epoch_{self.current_epoch+1}.pth'
            history_path = save_dir / f'training_history_epoch_{self.current_epoch+1}.json'
        
        # Save model with additional info
        additional_info = {
            'epoch': self.current_epoch,
            'best_score': self.best_score,
            'training_history': self.training_history,
            'config': self.config
        }
        
        save_method = getattr(self.model, 'save_model', None)
        if save_method is not None and callable(save_method):
            save_method(str(model_path), additional_info)
        else:
            # For models without save_model method, save using torch
            torch.save({
                'model_state_dict': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
                'additional_info': additional_info
            }, str(model_path))
        
        # Save training history
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"  Model saved to: {model_path}")
    
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate model performance."""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
        
        if data_loader is None:
            if self.test_loader is None:
                raise RuntimeError("Test data not initialized. Call setup_data() first.")
            data_loader = self.test_loader
        
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        print("Evaluating model...")
        pbar = tqdm(data_loader, desc='Evaluation', ncols=100)
        
        with torch.no_grad():
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Skip batch if contains NaN
                if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                    continue
                
                predictions = self.model(batch_X)
                loss = self._calculate_loss(predictions, batch_y, batch_X)
                
                # Skip if loss is NaN or inf
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                total_loss += loss.item()
                batch_count += 1
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / max(batch_count, 1)
        
        results = {
            'average_loss': avg_loss,
            'total_batches': batch_count,
            'mode': self.config['mode']
        }
        
        print(f"Evaluation Results:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Total Batches: {batch_count}")
        print(f"  Mode: {self.config['mode']}")
        
        return results
    
    def walk_forward_cv(self, n_splits: int = 5, min_train_size: int = 1000, 
                       step_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform walk-forward cross-validation for time series data.
        
        Args:
            n_splits: Number of splits for cross-validation
            min_train_size: Minimum size of training set
            step_size: Step size for walk-forward (defaults to test_size)
            
        Returns:
            Dictionary with cross-validation results
        """
        print(f"\nStarting walk-forward cross-validation with {n_splits} splits...")
        
        # Ensure data is loaded
        if self.dataset is None:
            self.setup_data()
        
        # Get the full dataset
        if self.dataset is None:
            raise RuntimeError("Dataset not properly initialized")
        
        if self.dataset.train_data is None or self.dataset.val_data is None or self.dataset.test_data is None:
            raise RuntimeError("Dataset components not properly initialized")
        
        X_full, y_full = self.dataset.train_data
        X_val, y_val = self.dataset.val_data
        X_test, y_test = self.dataset.test_data
        
        # Combine training and validation data for CV
        X_combined = np.concatenate([X_full, X_val], axis=0)
        y_combined = np.concatenate([y_full, y_val], axis=0)
        
        total_samples = len(X_combined)
        
        # Calculate split parameters
        if step_size is None:
            step_size = (total_samples - min_train_size) // n_splits
        
        cv_results = []
        fold_metrics = []
        
        print(f"Total samples: {total_samples}")
        print(f"Min train size: {min_train_size}")
        print(f"Step size: {step_size}")
        
        for fold in range(n_splits):
            print(f"\n=== Fold {fold + 1}/{n_splits} ===")
            
            # Define train/validation split for this fold
            train_end = min_train_size + fold * step_size
            val_start = train_end
            val_end = min(val_start + step_size, total_samples)
            
            if val_end <= val_start:
                print(f"Skipping fold {fold + 1}: insufficient data")
                continue
            
            # Split data for this fold
            X_train_fold = X_combined[:train_end]
            y_train_fold = y_combined[:train_end]
            X_val_fold = X_combined[val_start:val_end]
            y_val_fold = y_combined[val_start:val_end]
            
            print(f"Train: {len(X_train_fold)} samples")
            print(f"Val: {len(X_val_fold)} samples")
            
            # Create data loaders for this fold
            train_dataset = MemoryEfficientDataset(X_train_fold, y_train_fold)
            val_dataset = MemoryEfficientDataset(X_val_fold, y_val_fold)
            
            batch_size = min(512, 256)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                    num_workers=min(4, 2), pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=min(4, 2), pin_memory=True)
            
            # Reset model for this fold
            self.setup_model()
            self.setup_training()
            
            # Override data loaders for this fold
            self.train_loader = train_loader
            self.val_loader = val_loader
            
            # Train model for this fold
            fold_best_score = float('inf')
            fold_history = {'train_losses': [], 'val_scores': []}
            
            # Training loop for this fold
            early_stopping = EarlyStopping(patience=self.config['patience'])
            
            for epoch in range(self.config['epochs']):
                self.current_epoch = epoch
                
                # Train epoch
                train_loss = self.train_epoch()
                
                # Validate epoch
                val_loss = self.validate_epoch()
                
                # Update history
                fold_history['train_losses'].append(train_loss)
                fold_history['val_scores'].append(val_loss)
                
                # Track best score for this fold
                if val_loss < fold_best_score:
                    fold_best_score = val_loss
                
                # Print progress
                if epoch % 5 == 0 or epoch == self.config['epochs'] - 1:
                    print(f"  Epoch {epoch+1}/{self.config['epochs']}: "
                          f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                
                # Early stopping
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            # Store fold results
            fold_result = {
                'fold': fold + 1,
                'train_size': len(X_train_fold),
                'val_size': len(X_val_fold),
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'best_val_loss': fold_best_score,
                'final_train_loss': fold_history['train_losses'][-1],
                'epochs_trained': len(fold_history['train_losses']),
                'history': fold_history
            }
            
            cv_results.append(fold_result)
            fold_metrics.append(fold_best_score)
            
            print(f"  Fold {fold + 1} completed: Best Val Loss = {fold_best_score:.4f}")
        
        # Calculate cross-validation metrics
        cv_mean = np.mean(fold_metrics)
        cv_std = np.std(fold_metrics)
        
        cv_summary = {
            'n_splits': len(cv_results),
            'cv_mean_loss': cv_mean,
            'cv_std_loss': cv_std,
            'cv_scores': fold_metrics,
            'fold_results': cv_results
        }
        
        print(f"\n=== Walk-Forward CV Results ===")
        print(f"CV Mean Loss: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"Individual fold scores: {[f'{score:.4f}' for score in fold_metrics]}")
        
        return cv_summary
    
    def run_walk_forward_cv_pipeline(self, n_splits: int = 5, min_train_size: int = 1000, 
                                   step_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete walk-forward cross-validation pipeline.
        
        Args:
            n_splits: Number of splits for cross-validation
            min_train_size: Minimum size of training set
            step_size: Step size for walk-forward (defaults to test_size)
            
        Returns:
            Dictionary with cross-validation results
        """
        try:
            print("Starting walk-forward cross-validation pipeline...")
            
            # Setup data
            self.setup_data()
            
            # Run cross-validation
            cv_results = self.walk_forward_cv(n_splits, min_train_size, step_size)
            
            # After CV, train final model on full training data
            print("\nTraining final model on full training data...")
            self.setup_model()
            self.setup_training()
            self.train()
            
            # Evaluate final model on test set
            print("\nEvaluating final model on test set...")
            test_results = self.evaluate()
            
            # Combine results
            final_results = {
                'cross_validation': cv_results,
                'final_test_results': test_results
            }
            
            # Save results
            results_path = Path(self.config['save_dir']) / 'walk_forward_cv_results.json'
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            print(f"Walk-forward CV pipeline completed successfully!")
            print(f"Results saved to: {results_path}")
            
            return final_results
            
        except Exception as e:
            print(f"Walk-forward CV pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        try:
            print("Starting full training pipeline...")
            
            # Setup all components
            self.setup_data()
            self.setup_model()
            self.setup_training()
            
            # Train model
            self.train()
            
            # Evaluate on test set
            print("\nEvaluating on test set...")
            test_results = self.evaluate()
            
            # Save final results
            results_path = Path(self.config['save_dir']) / 'final_results.json'
            with open(results_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            print(f"Pipeline completed successfully!")
            print(f"Final results saved to: {results_path}")
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Modular Bitcoin Price Prediction Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'test', 'walk_forward_cv'],
                        default='walk_forward_cv', help='Mode to run')
    parser.add_argument('--model_path', type=str, help='Path to model for evaluation')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of splits for walk-forward cross-validation')
    parser.add_argument('--min_train_size', type=int, default=150000,
                        help='Minimum training size for walk-forward CV')
    parser.add_argument('--step_size', type=int, default=None,
                        help='Step size for walk-forward CV (defaults to auto-calculated)')
    
    args = parser.parse_args()
    
    # Check if configuration file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Available configurations:")
        config_loader = ConfigLoader()
        for config_file in config_loader.list_configs():
            print(f"  - {config_file}")
        return
    
    # Create trainer
    trainer = ModularTrainer(args.config)
    
    if args.mode == 'train':
        # Run full training pipeline
        success = trainer.run_full_pipeline()
        if not success:
            sys.exit(1)
    
    elif args.mode == 'evaluate':
        # Evaluate existing model
        if not args.model_path:
            print("Error: --model_path required for evaluation mode")
            return
        
        # Load model
        from predictors import load_predictor
        model = load_predictor(args.model_path)
        trainer.model = model.to(trainer.device)
        
        # Setup data
        trainer.setup_data()
        
        # Evaluate
        results = trainer.evaluate()
        print(f"Evaluation completed: {results}")
    
    elif args.mode == 'walk_forward_cv':
        # Run walk-forward cross-validation
        print("Running walk-forward cross-validation...")
        cv_results = trainer.run_walk_forward_cv_pipeline(
            n_splits=args.n_splits,
            min_train_size=args.min_train_size,
            step_size=args.step_size
        )
        if not cv_results:
            sys.exit(1)
    
    elif args.mode == 'test':
        # Test configuration loading
        print("Testing configuration loading...")
        print("Configuration loaded successfully!")
        
        # Print available predictors
        PredictorFactory.print_available_predictors()


if __name__ == "__main__":
    main() 