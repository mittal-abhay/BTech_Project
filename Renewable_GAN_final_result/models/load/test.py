# predict_load.py
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from models.load_model.train import ImprovedGenerator

# Load dataset
data = pd.read_csv("merged_dataset.csv")
data['date_time'] = pd.to_datetime(data['date_time'])

# Add time-based & lag features
data['hour'] = data['date_time'].dt.hour
data['day_of_week'] = data['date_time'].dt.dayofweek
data['month'] = data['date_time'].dt.month
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

for lag in [1, 24, 48, 72]:
    data[f'load_lag_{lag}'] = data['LOAD'].shift(lag)
data.dropna(inplace=True)

# Prepare input features
features_for_load = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity', 
                     'nwp_windspeed', 'lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature', 
                     'lmd_pressure', 'hour', 'day_of_week', 'month', 'is_weekend', 
                     'load_lag_1', 'load_lag_24', 'load_lag_48', 'load_lag_72']

scaler_features = StandardScaler()
scaled_features = scaler_features.fit_transform(data[features_for_load])
scaler_load = StandardScaler()
scaled_target = scaler_load.fit_transform(data[['LOAD']])

# Predict using Generator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = ImprovedGenerator(input_features=17, noise_dim=16, hidden_dim=128).to(device)
generator.load_state_dict(torch.load('best_LOAD_generator.pth'))
generator.eval()

test_tensor = torch.tensor(scaled_features, dtype=torch.float32)
test_dataset = TensorDataset(test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

predicted_load_list = []
with torch.no_grad():
    for batch in test_loader:
        batch_features = batch[0].to(device)
        noise = torch.randn(batch_features.size(0), generator.noise_dim, device=device)
        pred = generator(batch_features, noise)
        predicted_load_list.append(pred.cpu().numpy())

predicted_load = np.concatenate(predicted_load_list).flatten()
predicted_load_original = scaler_load.inverse_transform(predicted_load.reshape(-1, 1)).flatten()
data['predicted_load'] = predicted_load_original

# Save to CSV for next file
data.to_csv("dataset_with_predicted_load.csv", index=False)
print("Load predictions saved to dataset_with_predicted_load.csv")
