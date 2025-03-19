import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from train import Generator
from sklearn.metrics import r2_score
import matplotlib.dates as mdates

def test_model(test_data_path, generator, device, batch_size=32, num_samples=5):
    """
    Test the trained generator model on new data with multiple stochastic outputs.
    
    Parameters:
    test_data_path (str): Path to the test dataset CSV file
    generator (nn.Module): Trained generator model
    device (torch.device): Device to run the model on
    batch_size (int): Batch size for testing
    num_samples (int): Number of diverse samples to generate
    
    Returns:
    tuple: List of predicted samples, actual power values, and MAE of sample 1
    """
    # Load test dataset
    test_data = pd.read_csv(test_data_path)
    test_data['date_time'] = pd.to_datetime(test_data['date_time'], format="%Y-%m-%d %H:%M:%S")

    # # Read only from 01/12/21 to 30/12/21
    test_data = test_data[test_data['date_time'] >= '2021-12-01 00:00:00']
    test_data = test_data[test_data['date_time'] <= '2021-12-30 23:45:00']
    
    # Extract features
    test_features = test_data[['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 
                              'nwp_humidity', 'nwp_windspeed', 'lmd_totalirrad', 
                              'lmd_diffuseirrad', 'lmd_temperature', 'lmd_pressure']]
    test_power = test_data[['power']]
    
    # Normalize features and target
    scaler_features = MinMaxScaler()
    scaler_power = MinMaxScaler()
    scaled_test_features = scaler_features.fit_transform(test_features)
    scaled_test_power = scaler_power.fit_transform(test_power)
    
    # Convert to tensors
    test_features_tensor = torch.tensor(scaled_test_features, dtype=torch.float32)
    test_power_tensor = torch.tensor(scaled_test_power, dtype=torch.float32)
    
    # DataLoader
    test_dataset = TensorDataset(test_features_tensor, test_power_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate multiple stochastic outputs
    generator.eval()
    predicted_powers_list = []
    actual_power = []

    with torch.no_grad():
        for i in range(num_samples):
            predicted_power = []
            for batch_features, batch_power in test_loader:
                batch_features = batch_features.to(device)
                batch_power = batch_power.to(device)
                noise = torch.randn(batch_features.size(0), generator.noise_dim, device=device)
                predictions = generator(batch_features, noise)
                predicted_power.extend(predictions.cpu().numpy())
            predicted_powers_list.append(np.array(predicted_power))

        # Collect actual power once
        for batch_features, batch_power in test_loader:
            actual_power.extend(batch_power.cpu().numpy())
    
    actual_power = np.array(actual_power)

    # Calculate MAE on sample 1
    mae = np.mean(np.abs(predicted_powers_list[0] - actual_power))
    print(f"Mean Absolute Error (MAE) on test set (sample 1): {mae:.4f}")

    # Select date_time for the test set
    val_date_time = test_data['date_time'].iloc[len(test_data) - len(actual_power):]

    for i, predicted_power in enumerate(predicted_powers_list):
        plt.figure(figsize=(10, 5))
        plt.plot(val_date_time[:200], actual_power[:200], label="Actual Power", color="black", linewidth=2)
        plt.plot(val_date_time[:200], predicted_power[:200], label=f"Generated Power Sample {i+1}", color='red', alpha=0.8)
        
        plt.xlabel("Date Time")
        plt.ylabel("Power Output")
        plt.legend()
        plt.title(f"Actual vs Generated Power Output - Sample {i+1}")

        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))

        plt.tight_layout()
        plt.savefig(f"test_generated_sample_{i+1}.png", dpi=600)
        plt.show()

    return predicted_powers_list, actual_power, mae


# Usage example
if __name__ == "__main__":
    # Set device
    gpu_index = 6  # Use the same GPU index as training
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained generator
    generator = Generator(input_features=9).to(device)
    generator.load_state_dict(torch.load('generator_wgangp.pth'))
    
    # Test the model
    test_data_path = "testing_dataset/sing21new.csv"
    
    predicted_power, actual_power, mae = test_model(
        test_data_path=test_data_path,
        generator=generator,
        device=device
    )
    
