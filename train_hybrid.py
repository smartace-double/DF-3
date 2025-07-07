import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from pytorch_forecasting import TemporalFusionTransformer

class UnifiedPredictionModel(nn.Module):
    def __init__(self, ts_model_params, num_adapters=4):
        super().__init__()
        
        # Time Series Branch
        self.ts_model = TemporalFusionTransformer(**ts_model_params)
        
        # Text Processing Branch with Adapters
        bert_config = BertConfig.from_pretrained("yiyanghkust/finbert-tone")
        self.bert = BertModel.from_pretrained("yiyanghkust/finbert-tone", config=bert_config)
        
        # Add Adapters
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bert_config.hidden_size, 128),
                nn.GELU(),
                nn.Linear(128, bert_config.hidden_size)
            ) for _ in range(num_adapters)
        ])
        
        # Fusion Layers
        self.point_head = nn.Linear(ts_model_params.hidden_size + bert_config.hidden_size, 1)
        self.interval_head = nn.Linear(ts_model_params.hidden_size + bert_config.hidden_size, 2)  # [lower, upper]
        
        # Competition-Specific Components
        self.volatility_proj = nn.Linear(ts_model_params.hidden_size, 1)
        
    def forward(self, ts_inputs, text_inputs):
        # Time Series Processing
        ts_features = self.ts_model(ts_inputs)
        vol_estimate = torch.sigmoid(self.volatility_proj(ts_features)) * 0.2 + 0.01  # 1-21% range
        
        # Text Processing with Adapters
        bert_outputs = self.bert(**text_inputs)
        hidden_states = bert_outputs.last_hidden_state
        
        # Apply adapters (multi-adapter ensemble)
        adapter_outputs = []
        for adapter in self.adapters:
            adapter_outputs.append(adapter(hidden_states[:, 0, :]))  # CLS token
        text_features = torch.mean(torch.stack(adapter_outputs), dim=0)
        
        # Combined Features
        combined = torch.cat([ts_features, text_features], dim=1)
        
        # Competition-Specific Outputs
        point_pred = self.point_head(combined).squeeze(-1)
        
        # Interval prediction with volatility scaling
        interval_params = torch.exp(self.interval_head(combined))  # Always positive
        lower = point_pred - interval_params[:, 0] * vol_estimate.squeeze(-1)
        upper = point_pred + interval_params[:, 1] * vol_estimate.squeeze(-1)
        
        return torch.stack([point_pred, lower, upper], dim=1)

class ChallengeLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha  # Weight between point and interval
        
    def forward(self, outputs, targets):
        # targets: [true_price, future_volatility]
        point_pred = outputs[:, 0]
        lower = outputs[:, 1]
        upper = outputs[:, 2]
        true_price = targets[:, 0]
        vol = targets[:, 1]
        
        # Point prediction component (normalized absolute error)
        point_error = torch.abs(point_pred - true_price) / (true_price + 1e-6)
        
        # Interval component (scoring system replica)
        width = upper - lower
        inclusion = (true_price >= lower) & (true_price <= upper)
        
        # Width factor (normalized by volatility)
        width_factor = width / (vol * 3.29)  # 3.29 ≈ 1.64*2 for 90% CI
        
        # Inclusion factor
        inclusion_factor = inclusion.float()
        
        # Combined interval score (higher is better)
        interval_score = inclusion_factor * (1 - width_factor.clamp(0, 1))
        
        # Final loss (lower is better)
        loss = self.alpha * point_error - (1 - self.alpha) * interval_score
        
        return loss.mean()

def train_unified_model(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.AdamW([
        {'params': model.ts_model.parameters(), 'lr': 1e-3},
        {'params': model.bert.parameters(), 'lr': 5e-5},
        {'params': model.adapters.parameters(), 'lr': 1e-4},
        {'params': [*model.point_head.parameters(), 
                   *model.interval_head.parameters(),
                   *model.volatility_proj.parameters()], 'lr': 3e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[1e-3, 5e-5, 1e-4, 3e-4],
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )
    
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = ChallengeLoss(alpha=0.65)  # Slightly favor point predictions
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            ts_inputs, text_inputs, targets = batch
            
            with torch.cuda.amp.autocast():
                outputs = model(ts_inputs, text_inputs)
                loss = loss_fn(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        # Validation
        model.eval()
        val_scores = []
        with torch.no_grad():
            for batch in val_loader:
                ts_inputs, text_inputs, targets = batch
                outputs = model(ts_inputs, text_inputs)
                score = calculate_competition_score(outputs, targets)
                val_scores.append(score)
        
        print(f"Epoch {epoch}: Val Score = {torch.mean(torch.stack(val_scores)):.4f}")

class BitcoinDataset(Dataset):
    def __init__(self, ts_data, text_data, tokenizer, window=12):
        self.ts_data = ts_data  # Shape: (n_samples, n_features)
        self.text_data = text_data  # List of text strings
        self.tokenizer = tokenizer
        self.window = window  # Lookback window in 5-min intervals
        
    def __len__(self):
        return len(self.ts_data) - self.window - 12  # Predict 1hr ahead (12 steps)
        
    def __getitem__(self, idx):
        # Time series inputs (last hour data)
        ts_input = self.ts_data[idx:idx+self.window]
        
        # Text inputs (aggregated from last hour)
        text = " ".join(self.text_data[idx:idx+self.window])
        text_input = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=256,
            return_tensors='pt'
        )
        
        # Target (price and volatility 1hr ahead)
        future_prices = self.ts_data[idx+self.window:idx+self.window+12, 0]  # Close price
        target_price = future_prices[-1]
        target_vol = future_prices.std()
        
        return ts_input, text_input, torch.tensor([target_price, target_vol], dtype=torch.float32)

def calibrate_intervals(model, calibration_loader):
    """Adjust interval outputs based on recent performance"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in calibration_loader:
            outputs = model(*batch[:2])
            errors.append((outputs[:,0] - batch[2][:,0]).abs() / batch[2][:,0])
    
    avg_error = torch.cat(errors).mean()
    # Adjust interval widths proportionally to recent error
    model.interval_head.weight.data *= (1 + avg_error.item()/2)
    model.interval_head.bias.data *= (1 + avg_error.item()/2)

def optimize_for_inference(model):
    model.eval()
    
    # Apply torch.compile for faster execution
    model = torch.compile(model, mode='max-autotune')
    
    # Convert adapters to half precision
    for adapter in model.adapters:
        adapter = adapter.half()
    
    # Use BetterTransformer
    from optimum.bettertransformer import BetterTransformer
    model.bert = BetterTransformer.transform(model.bert)
    
    return model