# Train LightGBM model in precog mode (point + interval predictions)
python train_modular.py --config config/lightgbm_precog.yaml

# Train LightGBM model in synth mode (detailed timestep predictions)  
python train_modular.py --config config/lightgbm_synth.yaml

# Train LSTM model in precog mode (point + interval predictions)
python train_modular.py --config config/lstm_precog.yaml

# Train LSTM model in synth mode (detailed timestep predictions)  
python train_modular.py --config config/lstm_synth.yaml

# Train TCN model in precog mode
python train_modular.py --config config/tcn_precog.yaml

# Train TFT model in synth mode
python train_modular.py --config config/tft_synth.yaml