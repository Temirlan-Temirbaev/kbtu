import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def add_features(df):
    # Time-based features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Location-based features
    df['dist_from_equator'] = abs(df['lat'])
    df['lat_lon_ratio'] = df['lat'] / df['lon']
    df['depth_ratio'] = df['depth'] / (df['lat'] ** 2 + df['lon'] ** 2) ** 0.5
    
    return df

# Add features to both train and test
train_data = add_features(train_data)
test_data = add_features(test_data)

# Separate features for different models
time_columns = ['year', 'month', 'day', 'hour', 'min', 'sec', 
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
location_columns = ['lat', 'lon', 'depth', 'dist_from_equator', 
                   'lat_lon_ratio', 'depth_ratio']

# Create corresponding aftershock column names
time_columns_as = [col + '_as' for col in ['year', 'month', 'day', 'hour', 'min', 'sec']]
location_columns_as = [col + '_as' for col in ['lat', 'lon', 'depth']]

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
    def __init__(self, input_size=10, hidden_size=128, num_layers=3, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size//2, 6)  # Only predict original time columns
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        predictions = self.fc2(out)
        return predictions

# Prepare data
sequence_length = 15  # Increased sequence length
batch_size = 64      # Increased batch size

# Scale the data
time_scaler = MinMaxScaler()  # Changed to MinMaxScaler for time series
location_scaler = StandardScaler()
magnitude_scaler = StandardScaler()

scaled_time = time_scaler.fit_transform(train_data[time_columns])
scaled_location = location_scaler.fit_transform(train_data[location_columns])
scaled_magnitude = magnitude_scaler.fit_transform(train_data[['class']].values)

# Split data for validation
train_idx, val_idx = train_test_split(np.arange(len(scaled_time)), test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_time_dataset = TimeSeriesDataset(pd.DataFrame(scaled_time[train_idx], columns=time_columns), sequence_length)
val_time_dataset = TimeSeriesDataset(pd.DataFrame(scaled_time[val_idx], columns=time_columns), sequence_length)

train_loader = DataLoader(train_time_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_time_dataset, batch_size=batch_size)

# Initialize models
time_model = LSTMPredictor()
location_models = {
    'lat': Ridge(alpha=0.1),
    'lon': Ridge(alpha=0.1),
    'depth': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
}
magnitude_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

# Train LSTM model
criterion = nn.L1Loss()  # Changed to L1Loss (MAE)
optimizer = torch.optim.Adam(time_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_val_loss = float('inf')
patience = 10
patience_counter = 0

num_epochs = 150
for epoch in range(num_epochs):
    time_model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = time_model(batch_x)
        loss = criterion(outputs, batch_y[:, :6])  # Only compare original time columns
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    time_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = time_model(batch_x)
            val_loss += criterion(outputs, batch_y[:, :6]).item()
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Train location models
for feature, model in location_models.items():
    idx = location_columns.index(feature)
    model.fit(train_data[location_columns], train_data[location_columns[idx]])

# Train magnitude model with more features
magnitude_features = location_columns + ['year', 'month', 'day']
magnitude_model.fit(train_data[magnitude_features], train_data['class'])

# Make predictions for test data
def prepare_sequence(data, scaler, sequence_length):
    scaled_data = scaler.transform(data)
    sequences = np.array([scaled_data[i:i+sequence_length] for i in range(len(scaled_data) - sequence_length + 1)])
    return torch.FloatTensor(sequences)

# Prepare test sequences
test_data_time = test_data[time_columns].copy()
test_sequences = prepare_sequence(test_data_time.values, time_scaler, sequence_length)

# Make predictions
time_model.eval()
with torch.no_grad():
    time_predictions = time_model(test_sequences[-1].unsqueeze(0))
    time_predictions = time_scaler.inverse_transform(
        np.concatenate([time_predictions.numpy(), np.zeros((1, len(time_columns)-6))], axis=1)
    )[:, :6]  # Only take the first 6 columns (original time columns)

# Location predictions using separate models
location_predictions = np.column_stack([
    model.predict(test_data[location_columns]) 
    for feature, model in location_models.items()
])

# Magnitude predictions with more features
magnitude_predictions = magnitude_model.predict(test_data[magnitude_features])

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

# Round time-related predictions to integers except seconds
for col in ['year_as', 'month_as', 'day_as', 'hour_as', 'min_as']:
    submission[col] = submission[col].round().astype(int)

# Save predictions
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv") 