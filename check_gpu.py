#!/usr/bin/env python3
"""
GPU Status and Performance Check Script
"""

import torch
import time
import numpy as np

def check_gpu_status():
    """Check GPU availability and status"""
    print("="*60)
    print("GPU STATUS CHECK")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-Processor Count: {props.multi_processor_count}")
        
        # Set device
        device = torch.device('cuda:0')
        print(f"\nUsing device: {device}")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).to(device)
            print(f"✅ GPU memory allocation successful")
            print(f"Test tensor shape: {test_tensor.shape}")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU memory allocation failed: {e}")
            return False
            
    else:
        print("❌ CUDA is not available")
        print("Training will use CPU (much slower)")
        return False
    
    return True

def test_training_speed():
    """Test training speed with a simple model"""
    print("\n" + "="*60)
    print("TRAINING SPEED TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("Skipping speed test - no GPU available")
        return
    
    device = torch.device('cuda:0')
    
    # Create a simple LSTM model similar to yours
    class SimpleLSTM(torch.nn.Module):
        def __init__(self, input_size=20, hidden_size=256, num_layers=3):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, 36)  # 36 outputs like your model
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1])
    
    model = SimpleLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    # Create synthetic data
    batch_size = 128
    seq_len = 24
    input_size = 20
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256, 512]
    
    print(f"Testing training speed with different batch sizes:")
    print(f"Sequence length: {seq_len}, Input size: {input_size}")
    print("-" * 50)
    
    for bs in batch_sizes:
        try:
            # Create data
            X = torch.randn(bs, seq_len, input_size).to(device)
            y = torch.randn(bs, 36).to(device)
            
            # Warm up
            for _ in range(5):
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            
            # Time the training
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(10):  # 10 iterations
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            samples_per_sec = bs / avg_time
            
            print(f"Batch size {bs:3d}: {avg_time:.4f}s per batch ({samples_per_sec:.0f} samples/sec)")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {bs:3d}: ❌ Out of memory")
                break
            else:
                print(f"Batch size {bs:3d}: ❌ Error - {e}")
    
    # Clean up
    del model, optimizer
    torch.cuda.empty_cache()

def check_memory_usage():
    """Check current memory usage"""
    print("\n" + "="*60)
    print("MEMORY USAGE")
    print("="*60)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Available: {total - reserved:.2f} GB")
        
        if allocated > 0:
            print(f"  Usage:     {allocated/total*100:.1f}%")
    else:
        print("No GPU memory to check")

if __name__ == "__main__":
    gpu_available = check_gpu_status()
    check_memory_usage()
    
    if gpu_available:
        test_training_speed()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if gpu_available:
        print("✅ GPU is available - training should be fast")
        print("📈 Optimizations applied:")
        print("  - Increased batch size to 128+ for GPU efficiency")
        print("  - Added num_workers=4 and pin_memory=True")
        print("  - Enabled cudnn.benchmark for faster convolutions")
        print("  - Reduced hyperparameter search space")
        print("  - More aggressive early stopping")
    else:
        print("❌ No GPU available - training will be very slow")
        print("💡 Consider:")
        print("  - Using a cloud GPU service")
        print("  - Reducing model complexity")
        print("  - Using smaller batch sizes")
        print("  - Reducing the number of trials") 