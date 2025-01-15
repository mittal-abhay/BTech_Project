import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the features and target
features_nwp = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity', 'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure']
features_lmd = ['lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature', 'lmd_pressure', 'lmd_winddirection', 'lmd_windspeed']
features = features_nwp + features_lmd
target = 'power'

# Load and preprocess the data
def load_and_preprocess_data(source_data_path, target_data_path):
    source_data = pd.read_csv(source_data_path)
    target_data = pd.read_csv(target_data_path)
    
    # Ensure the date_time column is properly formatted
    source_data['date_time'] = pd.to_datetime(source_data['date_time'])
    target_data['date_time'] = pd.to_datetime(target_data['date_time'])
    
    # Normalize the features
    scaler = StandardScaler()
    source_data[features] = scaler.fit_transform(source_data[features])
    target_data[features] = scaler.transform(target_data[features])
    
    return source_data, target_data

# Define the improved CEDAN model
def build_cedan_model(input_shape):
    inputs = keras.Input(shape=(input_shape,))
    
    # Feature extractor
    features = keras.layers.Dense(128, activation='relu')(inputs)
    features = keras.layers.Dropout(0.3)(features)  # Dropout to prevent overfitting
    features = keras.layers.Dense(64, activation='relu')(features)
    features = keras.layers.Dense(32, activation='relu')(features)
    
    # Power prediction
    power_output = keras.layers.Dense(1, name='power_output')(features)
    
    # Domain classification
    domain_output = keras.layers.Dense(16, activation='relu')(features)
    domain_output = keras.layers.Dense(1, activation='sigmoid', name='domain_output')(domain_output)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=[power_output, domain_output])
    
    return model

# Train the CEDAN model with early stopping
def train_cedan(source_data, target_data, features, target):
    X_source = source_data[features].values
    y_source = source_data[target].values
    X_target = target_data[features].values
    
    # Create domain labels (0 for source, 1 for target)
    domain_source = np.zeros((X_source.shape[0], 1))
    domain_target = np.ones((X_target.shape[0], 1))
    
    # Combine source and target data
    X_combined = np.vstack((X_source, X_target))
    domain_combined = np.vstack((domain_source, domain_target))
    
    # Create dummy y values for target (won't be used in training)
    y_dummy = np.zeros((X_target.shape[0], 1))
    y_combined = np.vstack((y_source.reshape(-1, 1), y_dummy))
    
    # Build and compile the model
    model = build_cedan_model(X_source.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Lower learning rate for stability
        loss={
            'power_output': 'mse',
            'domain_output': 'binary_crossentropy'
        },
        loss_weights={
            'power_output': 1.0,
            'domain_output': 0.1
        },
        metrics={
            'power_output': 'mae',  # Mean Absolute Error as a metric
        }
    )
    
    # Use early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_combined,
        {
            'power_output': y_combined,
            'domain_output': domain_combined
        },
        epochs=100,
        batch_size=16,  # Smaller batch size
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    return model

# Main function to run the entire process
def main():
    source_data_path = 'station00.csv'  # Replace with actual source data path
    target_data_path = 'station00.csv'  # Replace with actual target data path
    
    # Load and preprocess data
    source_data, target_data = load_and_preprocess_data(source_data_path, target_data_path)
    
    # Split source data into train and test
    train_data, test_data = train_test_split(source_data, test_size=0.2, random_state=42)
    
    # Train CEDAN model
    model = train_cedan(train_data, target_data, features, target)
    
    # Make predictions on test data
    X_test = test_data[features].values
    y_test = test_data[target].values
    y_pred, _ = model.predict(X_test)
    
    # Create a plot
    plt.figure(figsize=(12, 6))
    
    # To reduce clutter, plot the first 500 samples
    plt.plot(test_data['date_time'][:500], y_test[:500], label='True Power', color='blue')
    plt.plot(test_data['date_time'][:500], y_pred[:500], label='Predicted Power', color='red')
    
    plt.xlabel('Date and Time')
    plt.ylabel('Power Output')
    plt.title('CEDAN: Predicted vs True Power Output')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('cedan_power_output_prediction_plot.png')
    plt.close()
    
    print("Plot has been saved as 'cedan_power_output_prediction_plot.png'")

if __name__ == "__main__":
    main()
