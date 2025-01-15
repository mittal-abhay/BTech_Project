import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import seaborn as sns


def add_time_features(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['season'] = (df['month'] % 12 + 3) // 3
    return df

def add_lag_features(df, target, lag=3):
    for i in range(1, lag+1):
        df[f'{target}_lag_{i}'] = df[target].shift(i)
    return df

def preprocess_data(data, features, target='power', time_lag=3, scaler=None):
    data = add_time_features(data)
    data = add_lag_features(data, target, lag=time_lag)
    data = data.dropna()
    
    if scaler is None:
        scaler = MinMaxScaler()
        data[features + [target]] = scaler.fit_transform(data[features + [target]])
    else:
        data[features + [target]] = scaler.transform(data[features + [target]])
    
    X, y = [], []
    for i in range(len(data) - time_lag):
        X.append(data[features].iloc[i:i+time_lag].values)
        y.append(data[target].iloc[i+time_lag])
    
    return np.array(X), np.array(y), scaler

def build_improved_model(input_shape, lstm_units=128, dropout_rate=0.3):
    model = models.Sequential([
        layers.LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(dropout_rate),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(dropout_rate),
        layers.LSTM(32),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Load and preprocess data
file_path = 'station00.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

features = ['nwp_globalirrad', 'nwp_directirrad', 'nwp_temperature', 'nwp_humidity', 
            'nwp_windspeed', 'nwp_winddirection', 'nwp_pressure',
            'lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature', 'lmd_pressure', 
            'lmd_winddirection', 'lmd_windspeed',
            'hour', 'day_of_week', 'month', 'season']

X, y, scaler = preprocess_data(data, features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = build_improved_model(input_shape=(X.shape[1], X.shape[2]))
early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, 
                    validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)

# Plot results (limited data points)
plt.figure(figsize=(12, 6))
sample_size = 500  # Adjust this to show more or fewer data points
plt.plot(y_test[:sample_size], label='True PV Output', color='blue')
plt.plot(y_pred[:sample_size], label='Predicted PV Output', color='red')
plt.xlabel('Time')
plt.ylabel('PV Power Output')
plt.title('True vs Predicted PV Power Output (Test Sample)')
plt.legend()
plt.tight_layout()
plt.show()

# Print performance metrics
mse = np.mean((y_test - y_pred.flatten())**2)
mae = np.mean(np.abs(y_test - y_pred.flatten()))
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")