import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch.optim.lr_scheduler import ReduceLROnPlateau
import holidays
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
gpu_index = 0  # Change as needed
if torch.cuda.is_available():
    device = torch.device(f'cuda:{gpu_index}')
    print(f"Using GPU: {torch.cuda.get_device_name(gpu_index)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

# Load dataset
data = pd.read_csv("merged_dataset.csv")
data['date_time'] = pd.to_datetime(data['date_time'])

# Add time-based features
data['hour'] = data['date_time'].dt.hour
data['day_of_week'] = data['date_time'].dt.dayofweek
data['month'] = data['date_time'].dt.month
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

# Add Singapore holiday feature
sg_holidays = holidays.SG()  # Use Singapore holidays
data['is_holiday'] = data['date_time'].dt.date.apply(lambda x: x in sg_holidays).astype(int)

# Select features including new time-based features
features = data[['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity', 
                'nwp_windspeed', 'lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature', 
                'lmd_pressure', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday']]
                
load_output = data[['LOAD']]

# Use StandardScaler instead of MinMaxScaler
scaler_feat = StandardScaler()
scaler_load = StandardScaler()

scaled_features = scaler_feat.fit_transform(features)
scaled_load = scaler_load.fit_transform(load_output)

# Time series split instead of random split
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(scaled_features):
    features_train_l, features_val_l = scaled_features[train_index], scaled_features[test_index]
    load_train, load_val = scaled_load[train_index], scaled_load[test_index]
    # Use the last split
    
# Convert to PyTorch tensors and create DataLoaders
train_loader_l = DataLoader(
    TensorDataset(torch.tensor(features_train_l, dtype=torch.float32), 
                  torch.tensor(load_train, dtype=torch.float32)), 
    batch_size=64,  # Increased batch size
    shuffle=True
)

val_loader_l = DataLoader(
    TensorDataset(torch.tensor(features_val_l, dtype=torch.float32), 
                  torch.tensor(load_val, dtype=torch.float32)), 
    batch_size=64,  # Increased batch size
    shuffle=False
)

# Enhanced Generator with residual connections
class ImprovedGenerator(nn.Module):
    def __init__(self, input_features=18, noise_dim=16, hidden_dim=128):
        super(ImprovedGenerator, self).__init__()
        self.noise_dim = noise_dim
        
        # Feature processing
        self.feature_layer = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Noise processing
        self.noise_layer = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # First hidden layer
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Second hidden layer with residual connection
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Third hidden layer
        self.hidden3 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, noise):
        x_feat = self.feature_layer(x)
        x_noise = self.noise_layer(noise)
        combined = torch.cat((x_feat, x_noise), dim=1)
        
        # First hidden layer
        h1 = self.hidden1(combined)
        
        # Second hidden layer with residual connection
        h2 = self.hidden2(h1) + h1
        
        # Third hidden layer
        h3 = self.hidden3(h2)
        
        # Output
        return self.output_layer(h3)

# Improved Discriminator
class ImprovedDiscriminator(nn.Module):
    def __init__(self, input_features=17, hidden_dim=128):
        super(ImprovedDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features + 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        return self.model(x)

# Compute gradient penalty for WGAN-GP
def compute_gradient_penalty(D, real_samples, fake_samples, features):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(features, interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# Initialize weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Calculate evaluation metrics
def calculate_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # Mean Absolute Percentage Error (avoiding division by zero)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# Training function with early stopping and learning rate scheduling
def train_wgan_gp_with_early_stopping(generator, discriminator, train_loader, val_loader, 
                                      model_name, num_epochs=200, lr_g=0.0001, lr_d=0.00001, 
                                      lambda_gp=20, n_critic=5, patience=20):
    # Initialize losses
    g_losses, d_losses = [], []
    val_metrics = []
    best_model_metric = float('inf')
    patience_counter = 0
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9))
    
    # Learning rate schedulers
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=10, verbose=True)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=10, verbose=True)
    
    for epoch in range(num_epochs):
        # Training phase
        generator.train()
        discriminator.train()
        g_loss_epoch, d_loss_epoch = 0, 0
        
        for batch_features, batch_target in train_loader:
            batch_size = batch_features.size(0)
            batch_features, batch_target = batch_features.to(device), batch_target.to(device)

            # Train Discriminator
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                
                # Generate noise with gradually decreasing scale
                noise_scale = max(0.1, 0.5 * (1 - epoch/num_epochs))
                noise = torch.randn(batch_size, generator.noise_dim, device=device) * noise_scale
                
                fake_target = generator(batch_features, noise).detach()
                real_validity = discriminator(batch_features, batch_target)
                fake_validity = discriminator(batch_features, fake_target)
                
                # Gradient penalty
                gp = compute_gradient_penalty(discriminator, batch_target, fake_target, batch_features)
                
                # Discriminator loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
                d_loss.backward()
                optimizer_D.step()
                d_loss_epoch += d_loss.item()

            # Train Generator
            optimizer_G.zero_grad()
            
            # Fresh noise for generator
            noise = torch.randn(batch_size, generator.noise_dim, device=device) * noise_scale
            
            gen_target = generator(batch_features, noise)
            fake_validity = discriminator(batch_features, gen_target)
            
            # Generator loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()
            g_loss_epoch += g_loss.item()

        # Average losses for the epoch
        g_losses.append(g_loss_epoch / len(train_loader))
        d_losses.append(d_loss_epoch / (len(train_loader) * n_critic))
        
        # Validation phase
        generator.eval()
        val_predictions = []
        actual_values = []
        
        with torch.no_grad():
            for val_features, val_targets in val_loader:
                val_features = val_features.to(device)
                noise = torch.zeros(val_features.size(0), generator.noise_dim, device=device)
                fake_output = generator(val_features, noise).cpu().numpy()
                val_predictions.extend(fake_output.flatten())
                actual_values.extend(val_targets.numpy().flatten())
        
        # Calculate validation metrics
        metrics = calculate_metrics(actual_values, val_predictions)
        val_metrics.append(metrics)
        
        # Update learning rate based on validation RMSE
        scheduler_G.step(metrics['RMSE'])
        scheduler_D.step(metrics['RMSE'])
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"[{model_name}] Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_losses[-1]:.4f} | G Loss: {g_losses[-1]:.4f}")
            print(f"Validation Metrics: MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")
        
        # Early stopping check
        current_metric = metrics['RMSE']  # Using RMSE as the metric for early stopping
        if current_metric < best_model_metric:
            best_model_metric = current_metric
            # Save best model
            torch.save(generator.state_dict(), f'best_{model_name}_generator.pth')
            torch.save(discriminator.state_dict(), f'best_{model_name}_discriminator.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Return losses and validation metrics history
    return g_losses, d_losses, val_metrics

# Initialize models
print("Initializing models...")
input_dim = features.shape[1]  # Updated input dimension with new features
gen_load = ImprovedGenerator(input_features=input_dim, noise_dim=16, hidden_dim=128).to(device)
disc_load = ImprovedDiscriminator(input_features=input_dim, hidden_dim=128).to(device)

load_generator_path = 'best_LOAD_generator.pth'
load_discriminator_path = 'best_LOAD_discriminator.pth'

if os.path.exists(load_generator_path) and os.path.exists(load_discriminator_path):
    gen_load.load_state_dict(torch.load(load_generator_path))
    disc_load.load_state_dict(torch.load(load_discriminator_path))
    print("Pre-trained models loaded successfully!")
else:
    print("Pre-trained models not found. Training from scratch...")
    # Apply weight initialization
    gen_load.apply(weights_init)
    disc_load.apply(weights_init)

    # Train models with early stopping
    print("Training Load GAN with early stopping...")
    g_losses_l, d_losses_l, val_metrics_l = train_wgan_gp_with_early_stopping(
        gen_load, disc_load, train_loader_l, val_loader_l, 
        model_name="LOAD", num_epochs=100, lr_g=0.0001, lr_d=0.0004, 
        lambda_gp=10, n_critic=5, patience=30
    )

    print("Training completed!")

    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(g_losses_l, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(d_losses_l, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('gan_losses_load.png', dpi=300)
    plt.close()


    gen_load.load_state_dict(torch.load('best_LOAD_generator.pth'))
    gen_load.eval()

    # Get predictions from the validation set
    load_predictions = []
    actual_load = []

    with torch.no_grad():
        for real_features, real_targets in val_loader_l:
            real_features = real_features.to(device)
            # Use zero noise for deterministic predictions
            noise = torch.zeros(real_features.size(0), gen_load.noise_dim, device=device)
            fake_load = gen_load(real_features, noise).cpu().numpy()
            # Also make ensemble predictions with different noise vectors
            ensemble_preds = []
            for _ in range(10):
                noise = torch.randn(real_features.size(0), gen_load.noise_dim, device=device) * 0.1
                pred = gen_load(real_features, noise).cpu().numpy()
                ensemble_preds.append(pred)
            
            # Average ensemble predictions
            ensemble_pred = np.mean(np.array(ensemble_preds), axis=0)
            
            load_predictions.extend(fake_load.flatten())
            actual_load.extend(real_targets.numpy().flatten())

    # Inverse transform predictions and actual values
    load_predictions_original = scaler_load.inverse_transform(np.array(load_predictions).reshape(-1, 1)).flatten()
    actual_load_original = scaler_load.inverse_transform(np.array(actual_load).reshape(-1, 1)).flatten()

    # Get corresponding dates for validation set
    val_dates = data['date_time'].iloc[-len(actual_load):]

    # Plot results
    plt.figure(figsize=(15, 6))
    plt.plot(val_dates[:100], actual_load_original[:100], label='Actual Load', color='black')
    plt.plot(val_dates[:100], load_predictions_original[:100], label='Generated Load', color='red', linestyle='--')
    plt.xlabel('Date Time')
    plt.ylabel('Load')
    plt.title('Actual vs Generated Load (Validation)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.tight_layout()
    plt.savefig('actual_vs_generated_load_.png', dpi=600)
    plt.close()

    # Calculate final metrics on the validation set
    final_metrics = calculate_metrics(actual_load_original, load_predictions_original)
    print("Final Validation Metrics:")
    print(f"MAE: {final_metrics['MAE']:.4f}")
    print(f"RMSE: {final_metrics['RMSE']:.4f}")
    print(f"MAPE: {final_metrics['MAPE']:.2f}%")
