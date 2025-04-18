import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from models.pv.train import Generator


def pv_model(test_data_path, batch_size=32):
    # Set device
    gpu_index = 6  # Use the same GPU index as training or else cpu if not available
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load the trained generator
    generator = Generator(input_features=9).to(device)
    generator.load_state_dict(torch.load('models/pv/saved_models/generator_wgangp.pth', map_location=device))
    
    # Load test dataset
    test_data = pd.read_csv(test_data_path)
    test_data['date_time'] = pd.to_datetime(test_data['date_time'], format="%Y-%m-%d %H:%M:%S")

    # Extract features
    test_features = test_data[['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 
                              'nwp_humidity', 'nwp_windspeed', 'lmd_totalirrad', 
                              'lmd_diffuseirrad', 'lmd_temperature', 'lmd_pressure']]
    # test_power = test_data[['power']]
    
    # Normalize features and target
    scaler_features = MinMaxScaler()
    # scaler_power = MinMaxScaler()
    scaled_test_features = scaler_features.fit_transform(test_features)
    # scaled_test_power = scaler_power.fit_transform(test_power)
    
    # Convert to tensors
    test_features_tensor = torch.tensor(scaled_test_features, dtype=torch.float32)
    # test_power_tensor = torch.tensor(scaled_test_power, dtype=torch.float32)
    
    # DataLoader
    # test_dataset = TensorDataset(test_features_tensor, test_power_tensor)
    test_dataset = TensorDataset(test_features_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate multiple stochastic outputs
    generator.eval()

    with torch.no_grad():
        predicted_power = []
        for batch_features in test_loader:
            batch_features = batch_features[0].to(device)
            noise = torch.randn(batch_features.size(0), generator.noise_dim, device=device)
            predictions = generator(batch_features, noise)
            predicted_power.extend(predictions.cpu().numpy())

    predicted_power = np.array(predicted_power).squeeze() 
    test_data['predicted_pv'] = predicted_power

    return test_data




    
