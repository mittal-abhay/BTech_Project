# predict_usep.py
import pandas as pd
import numpy as np
import holidays
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

data = pd.read_csv("test_data_with_predicted_pv.csv")
data['date_time'] = pd.to_datetime(data['date_time'])
data['is_holiday'] = data['date_time'].dt.date.apply(lambda x: x in holidays.SG()).astype(int)

# Add rolling features
data['rolling_mean_24'] = data['LOAD'].rolling(window=24).mean()
data['rolling_std_24'] = data['LOAD'].rolling(window=24).std()
data.dropna(inplace=True)

# Cyclical features
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# Final features for USEP model
usep_features = ['predicted_load', 'nwp_temperature', 'nwp_humidity', 'nwp_windspeed',
                 'rolling_mean_24', 'rolling_std_24', 'hour_sin', 'hour_cos', 
                 'month_sin', 'month_cos', 'is_weekend', 'is_holiday']

X = data[usep_features]
y = data['USEP']

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
xgb_model.fit(X, y)

# Predict
usep_pred = xgb_model.predict(X)

# Save to CSV
data['predicted_usep'] = usep_pred
data.to_csv("final_test_data.csv", index=False)
print("USEP predictions saved to final_test_data.csv")

# Metrics
rmse = mean_squared_error(y, usep_pred, squared=False)
r2 = r2_score(y, usep_pred)
print(f'RMSE: {rmse:.2f}, R2 Score: {r2:.4f}')

# Plot
plt.figure(figsize=(14,6))
plt.plot(data['date_time'][:100], y[:100], label='Actual USEP', marker='o')
plt.plot(data['date_time'][:100], usep_pred[:100], label='Predicted USEP', marker='x')
plt.legend()
plt.xlabel('Date Time')
plt.ylabel('USEP Price')
plt.title('Actual vs Predicted USEP (XGBoost)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('usep_pred.png')

