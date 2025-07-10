"""
Optuna Hyperparameter Optimization

This module provides comprehensive hyperparameter optimization using Optuna
for all Bitcoin price prediction models with dynamic optimizer and scheduler selection.
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Callable
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime
import warnings

# Import our framework components
from .cross_validation import WalkForwardCrossValidator, run_walk_forward_cv
from predictors import create_predictor_from_config
from losses import precog_loss, synth_loss
from preprocessing import BitcoinPreprocessor

warnings.filterwarnings('ignore')

class OptunaOptimizer:
    """
    Comprehensive Optuna-based hyperparameter optimization for Bitcoin price prediction.
    
    Supports dynamic optimization of model architecture, training parameters,
    optimizer selection, and learning rate scheduling.
    """
    
    def __init__(self, 
                 base_config: Dict[str, Any],
                 study_name: str = "bitcoin_prediction_study",
                 storage_url: Optional[str] = None,
                 n_trials: int = 100,
                 timeout: Optional[int] = None,
                 n_cv_folds: int = 5,
                 save_dir: str = "models/optuna",
                 verbose: bool = True):
        """
        Initialize the Optuna optimizer.
        
        Args:
            base_config: Base configuration dictionary
            study_name: Name for the Optuna study
            storage_url: Optional storage URL for resumable studies
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            n_cv_folds: Number of cross-validation folds
            save_dir: Directory to save results
            verbose: Whether to print detailed progress
        """
        self.base_config = base_config
        self.study_name = study_name
        self.storage_url = storage_url
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_cv_folds = n_cv_folds
        self.save_dir = Path(save_dir)
        self.verbose = verbose
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize study
        self._create_study()
        
        # Performance tracking
        self.best_trial_history = []
        self.completed_trials = 0
        
    def _create_study(self):
        """Create or load an Optuna study."""
        # Enhanced sampler for better exploration
        sampler = TPESampler(
            n_startup_trials=max(20, self.n_trials // 5),
            multivariate=True,
            group=True,
            constant_liar=True
        )
        
        # Aggressive pruning for efficiency
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=self.n_cv_folds,
            reduction_factor=3,
            bootstrap_count=5
        )
        
        # Create study
        if self.storage_url:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                sampler=sampler,
                pruner=pruner,
                direction='minimize',
                load_if_exists=True
            )
        else:
            self.study = optuna.create_study(
                sampler=sampler,
                pruner=pruner,
                direction='minimize'
            )
        
        if self.verbose:
            print(f"Optuna study created/loaded: {self.study_name}")
            print(f"  Previous trials: {len(self.study.trials)}")
            
    def suggest_architecture_params(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        """
        Suggest architecture-specific hyperparameters.
        
        Args:
            trial: Optuna trial object
            model_type: Type of model ('LSTM', 'LightGBM', 'TFT', 'TCN', 'GARCH')
            
        Returns:
            Dictionary of architecture parameters
        """
        params = {}
        
        if model_type.upper() == 'LSTM':
            params.update({
                'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
                'use_layer_norm': trial.suggest_categorical('use_layer_norm', [True, False]),
                'activation': trial.suggest_categorical('activation', ['SiLU', 'GELU', 'ReLU', 'Tanh']),
            })
            
        elif model_type.upper() == 'LIGHTGBM':
            params.update({
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            })
            
        elif model_type.upper() == 'TFT':
            params.update({
                'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
                'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
                'n_layers': trial.suggest_int('n_layers', 2, 6),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3),
                'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.3),
                'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
            })
            
        elif model_type.upper() == 'TCN':
            params.update({
                'num_channels': trial.suggest_categorical('num_channels', [[32, 64], [64, 128], [128, 256]]),
                'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                'activation': trial.suggest_categorical('activation', ['ReLU', 'GELU', 'SiLU']),
            })
            
        elif model_type.upper() == 'GARCH':
            params.update({
                'p': trial.suggest_int('p', 1, 5),
                'q': trial.suggest_int('q', 1, 5),
                'mean_model': trial.suggest_categorical('mean_model', ['Zero', 'Constant', 'AR']),
                'vol_model': trial.suggest_categorical('vol_model', ['GARCH', 'EGARCH', 'GJR-GARCH']),
                'dist': trial.suggest_categorical('dist', ['Normal', 'StudentT', 'SkewT']),
            })
            
        return params
    
    def suggest_training_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest training hyperparameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of training parameters
        """
        return {
            'epochs': trial.suggest_int('epochs', 10, 100),
            'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 2048]),
            'grad_clip': trial.suggest_float('grad_clip', 0.1, 2.0),
            'patience': trial.suggest_int('patience', 5, 20),
            'mixed_precision': trial.suggest_categorical('mixed_precision', [True, False]),
        }
    
    def suggest_optimizer_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest optimizer and its parameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of optimizer parameters
        """
        optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'RMSprop', 'SGD'])
        
        params = {
            'optimizer': optimizer_name,
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        }
        
        if optimizer_name == 'AdamW':
            params.update({
                'beta1': trial.suggest_float('beta1', 0.8, 0.99),
                'beta2': trial.suggest_float('beta2', 0.9, 0.999),
                'eps': trial.suggest_float('eps', 1e-8, 1e-6, log=True),
            })
        elif optimizer_name == 'Adam':
            params.update({
                'beta1': trial.suggest_float('beta1', 0.8, 0.99),
                'beta2': trial.suggest_float('beta2', 0.9, 0.999),
                'eps': trial.suggest_float('eps', 1e-8, 1e-6, log=True),
            })
        elif optimizer_name == 'RMSprop':
            params.update({
                'alpha': trial.suggest_float('alpha', 0.9, 0.999),
                'eps': trial.suggest_float('eps', 1e-8, 1e-6, log=True),
                'momentum': trial.suggest_float('momentum', 0.0, 0.9),
            })
        elif optimizer_name == 'SGD':
            params.update({
                'momentum': trial.suggest_float('momentum', 0.0, 0.9),
                'nesterov': trial.suggest_categorical('nesterov', [True, False]),
            })
        
        return params
    
    def suggest_scheduler_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest learning rate scheduler and its parameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of scheduler parameters
        """
        scheduler_name = trial.suggest_categorical('scheduler', [
            'OneCycleLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR', 'None'
        ])
        
        params = {'scheduler': scheduler_name}
        
        if scheduler_name == 'OneCycleLR':
            params.update({
                'pct_start': trial.suggest_float('pct_start', 0.1, 0.5),
                'div_factor': trial.suggest_float('div_factor', 10.0, 100.0),
                'final_div_factor': trial.suggest_float('final_div_factor', 100.0, 10000.0),
            })
        elif scheduler_name == 'CosineAnnealingLR':
            params.update({
                'T_max': trial.suggest_int('T_max', 10, 100),
                'eta_min': trial.suggest_float('eta_min', 1e-6, 1e-4, log=True),
            })
        elif scheduler_name == 'ReduceLROnPlateau':
            params.update({
                'factor': trial.suggest_float('factor', 0.1, 0.8),
                'patience': trial.suggest_int('patience', 5, 15),
                'min_lr': trial.suggest_float('min_lr', 1e-6, 1e-4, log=True),
            })
        elif scheduler_name == 'StepLR':
            params.update({
                'step_size': trial.suggest_int('step_size', 10, 50),
                'gamma': trial.suggest_float('gamma', 0.1, 0.9),
            })
        
        return params
    
    def suggest_preprocessing_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest preprocessing parameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of preprocessing parameters
        """
        return {
            'scaler_type': trial.suggest_categorical('scaler_type', ['standard', 'robust', 'minmax']),
            'lookback': trial.suggest_categorical('lookback', [48, 72, 96, 120]),
            'handle_outliers': trial.suggest_categorical('handle_outliers', [True, False]),
            'add_technical_features': trial.suggest_categorical('add_technical_features', [True, False]),
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value (lower is better)
        """
        try:
            # Get model type from base config
            model_type = self.base_config.get('model_type', 'LSTM')
            
            # Suggest all hyperparameters
            config = self.base_config.copy()
            config.update(self.suggest_architecture_params(trial, model_type))
            config.update(self.suggest_training_params(trial))
            config.update(self.suggest_optimizer_params(trial))
            config.update(self.suggest_scheduler_params(trial))
            config.update(self.suggest_preprocessing_params(trial))
            
            # Add trial info
            config['trial_number'] = trial.number
            config['study_name'] = self.study_name
            
            if self.verbose:
                print(f"\nTrial {trial.number}: Testing configuration")
                print(f"  Model: {model_type}")
                print(f"  Architecture: {self.suggest_architecture_params(trial, model_type)}")
                print(f"  Optimizer: {config['optimizer']}")
                print(f"  Scheduler: {config['scheduler']}")
            
            # Run cross-validation
            cv_results = run_walk_forward_cv(
                config=config,
                n_folds=self.n_cv_folds,
                save_models=False,  # Don't save models during optimization
                verbose=False  # Reduce verbosity during optimization
            )
            
            # Get the objective value
            objective_value = cv_results.mean_score
            
            # Store additional metrics as trial attributes
            trial.set_user_attr('cv_std', cv_results.std_score)
            trial.set_user_attr('best_fold', cv_results.best_fold)
            trial.set_user_attr('fold_scores', cv_results.fold_scores)
            trial.set_user_attr('config', config)
            
            # Update tracking
            self.completed_trials += 1
            
            if self.verbose:
                print(f"  Result: {objective_value:.4f} ± {cv_results.std_score:.4f}")
                
            # Report intermediate values for pruning
            for fold_idx, score in enumerate(cv_results.fold_scores):
                trial.report(score, fold_idx)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    if self.verbose:
                        print(f"  Trial {trial.number} pruned at fold {fold_idx}")
                    raise optuna.exceptions.TrialPruned()
            
            return objective_value
            
        except Exception as e:
            if self.verbose:
                print(f"  Trial {trial.number} failed: {str(e)}")
            return float('inf')
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Returns:
            Dictionary with optimization results
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Starting Optuna Optimization")
            print(f"{'='*80}")
            print(f"Study: {self.study_name}")
            print(f"Trials: {self.n_trials}")
            print(f"Timeout: {self.timeout}s" if self.timeout else "Timeout: None")
            print(f"CV Folds: {self.n_cv_folds}")
            print(f"Model: {self.base_config.get('model_type', 'LSTM')}")
            print(f"Mode: {self.base_config.get('mode', 'precog')}")
        
        # Add progress callback
        def progress_callback(study, trial):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Trial {trial.number} completed")
                if trial.value is not None:
                    print(f"  Score: {trial.value:.4f}")
                    if study.best_value is not None:
                        print(f"  Best: {study.best_value:.4f} (Trial {study.best_trial.number})")
                print(f"  Progress: {len(study.trials)}/{self.n_trials}")
                
                # Show parameter importance
                if len(study.trials) > 10:
                    try:
                        importance = optuna.importance.get_param_importances(study)
                        print("  Top parameters:")
                        for i, (param, imp) in enumerate(list(importance.items())[:5]):
                            print(f"    {param}: {imp:.3f}")
                    except:
                        pass
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=[progress_callback],
            gc_after_trial=True
        )
        
        # Get results
        results = self._get_optimization_results()
        
        # Save results
        self._save_results(results)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Optimization Completed")
            print(f"{'='*80}")
            print(f"Best Score: {results['best_value']:.4f}")
            print(f"Best Trial: {results['best_trial_number']}")
            print(f"Total Trials: {len(self.study.trials)}")
            print(f"Completed Trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
            
        return results
    
    def _get_optimization_results(self) -> Dict[str, Any]:
        """Get comprehensive optimization results."""
        best_trial = self.study.best_trial
        
        results = {
            'best_value': self.study.best_value,
            'best_trial_number': best_trial.number,
            'best_params': best_trial.params,
            'best_config': best_trial.user_attrs.get('config', {}),
            'cv_std': best_trial.user_attrs.get('cv_std', 0.0),
            'best_fold': best_trial.user_attrs.get('best_fold', 0),
            'fold_scores': best_trial.user_attrs.get('fold_scores', []),
            'study_name': self.study_name,
            'n_trials': len(self.study.trials),
            'optimization_time': datetime.now().isoformat(),
            'base_config': self.base_config,
        }
        
        # Add parameter importance
        if len(self.study.trials) > 10:
            try:
                results['param_importance'] = optuna.importance.get_param_importances(self.study)
            except:
                results['param_importance'] = {}
        
        # Add trial history
        results['trial_history'] = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                results['trial_history'].append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'duration': trial.duration.total_seconds() if trial.duration else None
                })
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results to disk."""
        # Save main results
        results_path = self.save_dir / f"{self.study_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save best configuration
        best_config_path = self.save_dir / f"{self.study_name}_best_config.json"
        with open(best_config_path, 'w') as f:
            json.dump(results['best_config'], f, indent=2)
        
        # Save study object
        study_path = self.save_dir / f"{self.study_name}_study.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        
        if self.verbose:
            print(f"Results saved to: {self.save_dir}")
    
    def load_study(self, study_path: str):
        """Load a previously saved study."""
        with open(study_path, 'rb') as f:
            self.study = pickle.load(f)
        
        if self.verbose:
            print(f"Study loaded from: {study_path}")
            print(f"  Trials: {len(self.study.trials)}")
            print(f"  Best value: {self.study.best_value}")


# Convenience function
def optimize_hyperparameters(base_config: Dict[str, Any],
                           study_name: str = "bitcoin_prediction",
                           n_trials: int = 100,
                           timeout: Optional[int] = None,
                           n_cv_folds: int = 5,
                           save_dir: str = "models/optuna",
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to optimize hyperparameters.
    
    Args:
        base_config: Base configuration dictionary
        study_name: Name for the study
        n_trials: Number of optimization trials
        timeout: Optional timeout in seconds
        n_cv_folds: Number of cross-validation folds
        save_dir: Directory to save results
        verbose: Whether to print progress
        
    Returns:
        Dictionary with optimization results
    """
    optimizer = OptunaOptimizer(
        base_config=base_config,
        study_name=study_name,
        n_trials=n_trials,
        timeout=timeout,
        n_cv_folds=n_cv_folds,
        save_dir=save_dir,
        verbose=verbose
    )
    
    return optimizer.optimize() 