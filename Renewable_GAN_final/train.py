import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# Set device
gpu_index = 5
if torch.cuda.is_available():
    device = torch.device(f'cuda:{gpu_index}')
    print(f"Using GPU: {torch.cuda.get_device_name(gpu_index)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

# Load dataset
data = pd.read_csv("training_dataset/station00.csv")
features = data[['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity', 
                 'nwp_windspeed', 'lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature', 'lmd_pressure']]
power_output = data[['power']]

# Normalize features and target
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_power = scaler.fit_transform(power_output)

# Train-Validation split
features_train, features_val, power_train, power_val = train_test_split(
    scaled_features, scaled_power, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_features_tensor = torch.tensor(features_train, dtype=torch.float32)
train_power_tensor = torch.tensor(power_train, dtype=torch.float32)
val_features_tensor = torch.tensor(features_val, dtype=torch.float32)
val_power_tensor = torch.tensor(power_val, dtype=torch.float32)

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_features_tensor, train_power_tensor)
val_dataset = TensorDataset(val_features_tensor, val_power_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class Generator(nn.Module):
    def __init__(self, input_features=9):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # First layer - expand features
            nn.Linear(input_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Hidden layers with skip connections
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_size=1):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training setup
def train_gan(generator, discriminator, train_loader, num_epochs, device, lr=0.0002, beta1=0.5):
    # Loss functions
    adversarial_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    # Optimizers with better parameters
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, verbose=True)
    
    g_losses, d_losses = [], []
    
    for epoch in range(num_epochs):
        g_loss_epoch, d_loss_epoch = 0, 0
        
        for batch_features, batch_power in train_loader:
            batch_size = batch_features.size(0)
            batch_features, batch_power = batch_features.to(device), batch_power.to(device)

            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # -----------------
            # Train Discriminator
            # -----------------
            optimizer_D.zero_grad()
            
            # Real power output
            d_real = discriminator(batch_power)
            d_real_loss = adversarial_loss(d_real, real_labels)
            
            # Generate fake power output
            fake_power = generator(batch_features)
            d_fake = discriminator(fake_power.detach())
            d_fake_loss = adversarial_loss(d_fake, fake_labels)
            
            # Combined D loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Try to fool the discriminator
            g_fake = discriminator(fake_power)
            g_fake_loss = adversarial_loss(g_fake, real_labels)
            
            # Add MSE loss to ensure output matches real values
            g_mse = mse_loss(fake_power, batch_power)
            
            # Combined G loss with weighted terms
            g_loss = 0.7 * g_mse + 0.3 * g_fake_loss
            g_loss.backward()
            optimizer_G.step()
            
            # Record losses
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            
        # Calculate average epoch losses
        g_loss_epoch /= len(train_loader)
        d_loss_epoch /= len(train_loader)
        
        # Update learning rates
        scheduler_G.step(g_loss_epoch)
        scheduler_D.step(d_loss_epoch)
        
        g_losses.append(g_loss_epoch)
        d_losses.append(d_loss_epoch)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss_epoch:.4f} | G Loss: {g_loss_epoch:.4f}")
    
    return g_losses, d_losses

# Usage
generator = Generator(input_features=9).to(device)
discriminator = Discriminator(input_size=1).to(device)

generator_path = 'generator.pth'
discriminator_path = 'discriminator.pth'

if os.path.exists(generator_path) and os.path.exists(discriminator_path):
    # Load the saved models
    generator.load_state_dict(torch.load(generator_path))
    discriminator.load_state_dict(torch.load(discriminator_path))
    print("Models loaded successfully, skipping training.")
else:
    # Train the model
    g_losses, d_losses = train_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        num_epochs=200,  # Increased epochs
        device=device,
        lr=0.0001  # Reduced learning rate
    )
    # Save the trained models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("Models saved successfully.")

    # Plot and save the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig('loss_plot.png')

# Evaluate and calculate accuracy
generator.eval()
predicted_power = []
actual_power = []
with torch.no_grad():
    for real_features, real_power in val_loader:
        real_features, real_power = real_features.to(device), real_power.to(device)
        fake_power = generator(real_features).cpu().numpy()
        actual_power.extend(real_power.cpu().numpy())
        predicted_power.extend(fake_power)

# Calculate Mean Absolute Error (MAE) for accuracy
predicted_power = np.array(predicted_power)
actual_power = np.array(actual_power)
mae = np.mean(np.abs(predicted_power - actual_power))
print(f"Mean Absolute Error (MAE) on validation set: {mae:.4f}")

# Plot and save the Actual vs Predicted Power output

# plt.figure(figsize=(10, 5))
# plt.plot(actual_power[:100], label="Actual Power")
# # show predicted power in red color
# plt.plot(predicted_power[:100], label="Predicted Power", color='red')
# plt.xlabel("Sample Index")
# plt.ylabel("Power Output")
# plt.legend()
# plt.title("Actual vs Predicted Power Output")
# plt.savefig("actual_vs_predicted_power.png")

# Plot and save the Actual vs Predicted Power output with date_time as x-axis

import matplotlib.dates as mdates

# Ensure date_time column is in datetime format
data['date_time'] = pd.to_datetime(data['date_time'])

# Select date_time for the validation set
val_date_time = data['date_time'].iloc[len(data) - len(actual_power):]

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
plt.savefig("actual_vs_predicted_power.png")
plt.show()





