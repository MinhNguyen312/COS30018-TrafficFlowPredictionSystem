import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load GRU Model
model = tf.keras.models.load_model('../model/gru.h5', 
                                 custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

# Load both training and test data
df_train = pd.read_csv('../data/2000_training_data/2000_flow_train.csv')
df_test = pd.read_csv('../data/2000_training_data/2000_flow_test.csv')

# Convert time columns
df_test['Time'] = pd.to_datetime(df_test['Time'])

# Get values
train_values = df_train['Lane Flow (Veh/15 Minutes)'].values
test_values = df_test['Lane Flow (Veh/15 Minutes)'].values

# Scale using training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_values.reshape(-1, 1))
test_values_scaled = scaler.transform(test_values.reshape(-1,1)).flatten()

# Create sequences
sequence_length = 12
X_test = []
for i in range(len(test_values_scaled) - sequence_length):
    X_test.append(test_values_scaled[i: i + sequence_length])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # 3D shape for GRU

# Make predictions
predicted_values_scaled = model.predict(X_test)

# Inverse transform predictions
predicted_values = scaler.inverse_transform(predicted_values_scaled)
actual_values = test_values[sequence_length:]

# Calculate error metrics
mse = mean_squared_error(actual_values, predicted_values)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_values, predicted_values)
mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

print(f'Test Metrics:')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df_test['Time'][sequence_length:], actual_values, 
         label="Actual", color='blue', alpha=0.7)
plt.plot(df_test['Time'][sequence_length:], predicted_values, 
         label="Predicted", color='red', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Lane Flow (Veh/15 Minutes)')
plt.title("Actual vs Predicted Lane Flow (GRU Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()