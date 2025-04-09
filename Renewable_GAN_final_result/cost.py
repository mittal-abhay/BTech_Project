import pandas as pd

data = pd.read_csv("final_test_data.csv")

# Predicted Cost feature calculated as USEP * max(0, (predicted_load - predicted_pv))
data['predicted_cost'] = data['predicted_usep'] * data[['predicted_load', 'predicted_pv']].apply(lambda x: max(0, x[0] - x[1]), axis=1)

# Cost feature calculated as USEP * max(0, (LOAD - PV))
data['cost'] = data['USEP'] * data[['LOAD', 'power']].apply(lambda x: max(0, x[0] - x[1]), axis=1)

# Now save only the relevant columns
data[['date_time', 'predicted_cost', 'cost']].to_csv("final_test_data_with_cost.csv", index=False)
print("Cost predictions saved to final_test_data_with_cost.csv")

# Plot cost and predicted cost for first 100 samples
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
plt.plot(data['date_time'][:100], data['cost'][:100], label='Actual Cost', marker='o')
plt.plot(data['date_time'][:100], data['predicted_cost'][:100], label='Predicted Cost', marker='x')
plt.legend()
plt.xlabel('Date Time')
plt.ylabel('Cost')
plt.title('Actual vs Predicted Cost')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cost_pred.png')

# Comparison metrics too
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(data['cost'], data['predicted_cost'], squared=False)
mae = mean_absolute_error(data['cost'], data['predicted_cost'])

print(f'RMSE: {rmse:.2f}, MAE: {mae:.2f}')
