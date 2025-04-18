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
import matplotlib.dates as mdates

# Set device
gpu_index = 6
if torch.cuda.is_available():
    device = torch.device(f'cuda:{gpu_index}')
    print(f"Using GPU: {torch.cuda.get_device_name(gpu_index)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

class Generator(nn.Module):
        def __init__(self, input_features=9, noise_dim=10):
            super(Generator, self).__init__()
            self.noise_dim = noise_dim
            self.model = nn.Sequential(
                nn.Linear(input_features + noise_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x, noise):
            x = torch.cat((x, noise), dim=1)
            return self.model(x)

class Discriminator(nn.Module):
        def __init__(self, input_features=9):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_features + 1, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid() 
        )

        def forward(self, x, power):
            x = torch.cat((x, power), dim=1)
            return self.model(x)
        
# If model already exists, load it
if os.path.exists('models/pv/saved_models/generator_wgangp.pth') and os.path.exists('models/pv/saved_models/discriminator_wgangp.pth'):
    print("Models loaded successfully.")
else:
    data = pd.read_csv("../../../data/training_dataset/station00.csv")
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



    # Gradient Penalty
    def compute_gradient_penalty(D, real_samples, fake_samples, features):
        alpha = torch.rand(real_samples.size(0), 1).to(device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
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

    # Weight Initialization
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    # Train loop adjustments
    def train_wgan_gp(generator, discriminator, train_loader, num_epochs, device, lr_g=0.00005, lr_d=0.0002, lambda_gp=10, n_critic=5):
        g_losses, d_losses = [], []

        optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9))

        for epoch in range(num_epochs):
            g_loss_epoch, d_loss_epoch = 0, 0

            for batch_features, batch_power in train_loader:
                batch_size = batch_features.size(0)
                batch_features, batch_power = batch_features.to(device), batch_power.to(device)

                # Train Discriminator n_critic times
                for _ in range(n_critic):
                    optimizer_D.zero_grad()
                    noise = torch.randn(batch_size, generator.noise_dim, device=device) * 0.5
                    fake_power = generator(batch_features, noise).detach()
                    real_validity = discriminator(batch_features, batch_power)
                    fake_validity = discriminator(batch_features, fake_power)
                    gp = compute_gradient_penalty(discriminator, batch_power, fake_power, batch_features)
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp
                    d_loss.backward()
                    optimizer_D.step()
                    d_loss_epoch += d_loss.item()

                # Train Generator once
                optimizer_G.zero_grad()
                noise = torch.randn(batch_size, generator.noise_dim, device=device) * 0.5
                gen_power = generator(batch_features, noise)
                g_loss = -torch.mean(discriminator(batch_features, gen_power))
                g_loss.backward()
                optimizer_G.step()
                g_loss_epoch += g_loss.item()

            g_loss_epoch /= len(train_loader)
            d_loss_epoch /= (len(train_loader) * n_critic)
            g_losses.append(g_loss_epoch)
            d_losses.append(d_loss_epoch)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss_epoch:.4f} | G Loss: {g_loss_epoch:.4f}")

        return g_losses, d_losses



    # Usage
    generator = Generator(input_features=9).to(device)
    discriminator = Discriminator(input_features=9).to(device)

    generator_path = 'generator_wgangp.pth'
    discriminator_path = 'discriminator_wgangp.pth'

   
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    g_losses, d_losses = train_wgan_gp(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        num_epochs=200,
        device=device,
        lr_g=0.0001,
        lr_d=0.0004  # Different LR
    )
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    print("Models saved successfully.")

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig('loss_plot_wgan_gp.png')


    # Evaluate with multiple samples due to noise
    generator.eval()
    num_samples = 5  # How many diverse outputs you want
    predicted_powers_list = []
    actual_power = []

    with torch.no_grad():
        for i in range(num_samples):
            predicted_power = []
            for real_features, real_power in val_loader:
                real_features, real_power = real_features.to(device), real_power.to(device)
                noise = torch.randn(real_features.size(0), generator.noise_dim, device=device)
                fake_power = generator(real_features, noise).cpu().numpy()
                predicted_power.extend(fake_power)
            predicted_powers_list.append(np.array(predicted_power))

        # Get actual power only once (from first pass)
        for real_features, real_power in val_loader:
            actual_power.extend(real_power.cpu().numpy())

    actual_power = np.array(actual_power)

    # Calculate MAE for one of the samples (optional)
    mae = np.mean(np.abs(predicted_powers_list[0] - actual_power))
    print(f"Mean Absolute Error (MAE) on validation set (sample 1): {mae:.4f}")

    # Time info
    data['date_time'] = pd.to_datetime(data['date_time'])
    val_date_time = data['date_time'].iloc[len(data) - len(actual_power):]

    # Plot Actual vs Multiple Predictions
    plt.figure(figsize=(12, 6))
    plt.plot(val_date_time[:100], actual_power[:100], label="Actual Power", color='black', linewidth=2)

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, predicted_power in enumerate(predicted_powers_list):
        plt.plot(val_date_time[:100], predicted_power[:100], label=f"Generated Power Sample {i+1}", color=colors[i], alpha=0.7)

    plt.xlabel("Date Time")
    plt.ylabel("Power Output")
    plt.legend()
    plt.title("Actual vs Multiple Generated Power Outputs (CGAN diversity)")

    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.tight_layout()
    plt.savefig("actual_vs_multiple_generated_power.png", dpi=600)
    plt.show()