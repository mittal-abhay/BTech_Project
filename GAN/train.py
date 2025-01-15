import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import RenewableWGAN
from load import prepare_data_for_gan
import tensorflow as tf

def plot_training_history(critic_losses, generator_losses):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(critic_losses, label='Critic Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Critic Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Training Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_predictions(test_dates, actual_power, predicted_power, num_samples=5):
    plt.figure(figsize=(15, 6))
    
    # Plot actual power
    plt.plot(test_dates, actual_power.flatten(), label="Actual Power", color="blue", alpha=0.7)
    
    # Plot a few predicted scenarios
    for i in range(min(num_samples, predicted_power.shape[0])):
        plt.plot(test_dates, predicted_power[i].flatten(), 
                label=f"Predicted Scenario {i+1}", 
                alpha=0.3, linestyle="--")
    
    plt.xlabel("Date-Time")
    plt.ylabel("Power Output")
    plt.title("Actual vs Predicted Power Output Scenarios")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Prepare data for GAN
    print("Preparing data...")
    filepath = "station00.csv"  
    sequences, features, df, scaler = prepare_data_for_gan(filepath)
    
    # Split the dataset
    print("Splitting dataset...")
    train_sequences, test_sequences = train_test_split(
        sequences, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Model parameters
    sequence_length = 24
    feature_dim = len(features)
    latent_dim = 100
    conditional_dim = 0  # No conditional inputs
    
    # Initialize the WGAN model
    print("Initializing WGAN model...")
    wgan = RenewableWGAN(
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        conditional_dim=conditional_dim,
        latent_dim=latent_dim
    )
    wgan.scaler = scaler  # Set the scaler
    
    # Training parameters
    epochs = 10
    batch_size = 32
    
    # Train the model
    print("Training the model...")
    try:
        critic_losses, generator_losses = wgan.train(
            train_sequences, epochs=epochs, batch_size=batch_size
        )
        
        # Plot training history
        plot_training_history(critic_losses, generator_losses)
        
        # Generate predictions
        print("Generating predictions...")
        num_scenarios = 10
        predicted_scenarios = wgan.generate_scenarios(num_scenarios)
        
        # Get corresponding test dates for plotting
        test_dates = df.index[-len(test_sequences):]
        
        # Plot predictions
        plot_predictions(test_dates, test_sequences, predicted_scenarios)
        
        print("Training and evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during training or evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
