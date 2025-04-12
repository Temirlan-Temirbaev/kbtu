import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from torch.utils.data import Dataset, DataLoader

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate features for different models
time_columns = ['year', 'month', 'day', 'hour', 'min', 'sec']
location_columns = ['lat', 'lon', 'depth']

# Create corresponding aftershock column names
time_columns_as = [col + '_as' for col in time_columns]
location_columns_as = [col + '_as' for col in location_columns]

# LSTM for time prediction
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        self.data = torch.FloatTensor(data[time_columns].values)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        return (self.data[idx:idx+self.sequence_length], 
                self.data[idx+self.sequence_length])

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

# Prepare data
sequence_length = 10
batch_size = 32

# Scale the data
time_scaler = StandardScaler()
location_scaler = StandardScaler()
magnitude_scaler = StandardScaler()

scaled_time = time_scaler.fit_transform(train_data[time_columns])
scaled_location = location_scaler.fit_transform(train_data[location_columns])
scaled_magnitude = magnitude_scaler.fit_transform(train_data[['class']].values)

# Create datasets and dataloaders
time_dataset = TimeSeriesDataset(pd.DataFrame(scaled_time, columns=time_columns), sequence_length)
time_loader = DataLoader(time_dataset, batch_size=batch_size, shuffle=True)

# Initialize models
time_model = LSTMPredictor()
location_model = LinearRegression()
magnitude_model = GradientBoostingRegressor()

# Train LSTM model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(time_model.parameters())

num_epochs = 100
for epoch in range(num_epochs):
    for batch_x, batch_y in time_loader:
        optimizer.zero_grad()
        outputs = time_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Train location model
location_model.fit(train_data[location_columns], train_data[location_columns])

# Train magnitude model
magnitude_model.fit(train_data[location_columns], train_data['class'])

# Make predictions for test data
def prepare_sequence(data, scaler, sequence_length):
    scaled_data = scaler.transform(data)
    sequences = np.array([scaled_data[i:i+sequence_length] for i in range(len(scaled_data) - sequence_length + 1)])
    return torch.FloatTensor(sequences)

# Prepare test sequences
test_sequences = prepare_sequence(test_data[time_columns].values, time_scaler, sequence_length)

# Make predictions
with torch.no_grad():
    time_predictions = time_model(test_sequences[-1].unsqueeze(0))
    time_predictions = time_scaler.inverse_transform(time_predictions.numpy())

location_predictions = location_model.predict(test_data[location_columns])
magnitude_predictions = magnitude_model.predict(test_data[location_columns])

# Prepare submission
submission = pd.read_csv('sample_submission.csv')

# Fill predictions
for i in range(len(submission)):
    # Fill time predictions
    for j, col in enumerate(time_columns_as):
        submission.loc[i, col] = time_predictions[0][j]
    
    # Fill location predictions
    for j, col in enumerate(location_columns_as):
        submission.loc[i, col] = location_predictions[i][j]
    
    # Fill magnitude prediction
    submission.loc[i, 'class_as'] = magnitude_predictions[i]

# Save predictions
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv") 