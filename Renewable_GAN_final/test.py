import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from train import Generator
from sklearn.metrics import r2_score
import matplotlib.dates as mdates

def test_model(test_data_path, generator, device, batch_size=32):
    """
    Test the trained generator model on new data
    
    Parameters:
    test_data_path (str): Path to the test dataset CSV file
    generator (nn.Module): Trained generator model
    device (torch.device): Device to run the model on
    batch_size (int): Batch size for testing
    
    Returns:
    tuple: Arrays of predicted and actual power values, and the test MAE
    """
    # Load test dataset
    test_data = pd.read_csv(test_data_path)
    
    # Extract features (using same columns as training)
    test_features = test_data[['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 
                              'nwp_humidity', 'nwp_windspeed', 'lmd_totalirrad', 
                              'lmd_diffuseirrad', 'lmd_temperature', 'lmd_pressure']]
    test_power = test_data[['power']]
    
    # Normalize features and target using the same scaling approach
    scaler_features = MinMaxScaler()
    scaler_power = MinMaxScaler()
    
    scaled_test_features = scaler_features.fit_transform(test_features)
    scaled_test_power = scaler_power.fit_transform(test_power)
    
    # Convert to PyTorch tensors
    test_features_tensor = torch.tensor(scaled_test_features, dtype=torch.float32)
    test_power_tensor = torch.tensor(scaled_test_power, dtype=torch.float32)
    
    # Create test dataset and dataloader
    test_dataset = TensorDataset(test_features_tensor, test_power_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate model
    generator.eval()
    predicted_power = []
    actual_power = []
    
    with torch.no_grad():
        for batch_features, batch_power in test_loader:
            batch_features = batch_features.to(device)
            batch_power = batch_power.to(device)
            
            # Generate predictions
            predictions = generator(batch_features)
            
            # Store predictions and actual values
            predicted_power.extend(predictions.cpu().numpy())
            actual_power.extend(batch_power.cpu().numpy())
    
    # Convert to numpy arrays
    predicted_power = np.array(predicted_power)
    actual_power = np.array(actual_power)
    
    # Calculate metrics
    mae = np.mean(np.abs(predicted_power - actual_power))
    mse = np.mean((predicted_power - actual_power) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_power, predicted_power)
    
    
    # Print metrics
    print(f"Test Set Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # # Plot results
    # plt.figure(figsize=(12, 6))
    # plt.plot(actual_power[:1000], label="Actual Power", alpha=0.7)
    # plt.plot(predicted_power[:1000], label="Predicted Power", color='red', alpha=0.7)
    # plt.xlabel("Sample Index")
    # plt.ylabel("Power Output")
    # plt.title("Actual vs Predicted Power Output (Test Set)")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("test_set_predictions.png")
    # plt.close()
    


    # Ensure date_time column is in datetime format
    test_data['date_time'] = pd.to_datetime(test_data['date_time'])

    # Select date_time for the validation set
    val_date_time = test_data['date_time'].iloc[len(test_data) - len(actual_power):]

    plt.figure(figsize=(10, 5))
    plt.plot(val_date_time[:100], actual_power[:100], label="Actual Power")
    plt.plot(val_date_time[:100], predicted_power[:100], label="Predicted Power", color='red')
    plt.xlabel("Date Time")
    plt.ylabel("Power Output")
    plt.legend()
    plt.title("Actual vs Predicted Power Output")

    # Formatting x-axis to show dates and rotate them for better readability
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Adjust interval as needed

    plt.tight_layout()
    plt.savefig("test_set_predictions.png")
    plt.show()

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_power, predicted_power, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.xlabel("Actual Power")
    plt.ylabel("Predicted Power")
    plt.title("Actual vs Predicted Power Scatter Plot")
    plt.grid(True)
    plt.savefig("test_set_scatter.png")
    plt.close()
    
    return predicted_power, actual_power, mae

# Usage example
if __name__ == "__main__":
    # Set device
    gpu_index = 5  # Use the same GPU index as training
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained generator
    generator = Generator(input_features=9).to(device)
    generator.load_state_dict(torch.load('generator.pth'))
    
    # Test the model
    test_data_path = "testing_dataset/station04.csv"  # Replace with your test data path
    predicted_power, actual_power, mae = test_model(
        test_data_path=test_data_path,
        generator=generator,
        device=device
    )
    