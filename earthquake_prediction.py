import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import warnings
from sklearn.svm import SVR
from torch.optim.lr_scheduler import ReduceLROnPlateau
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def add_features(df):
    # Existing features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['dist_from_equator'] = abs(df['lat'])
    df['lat_lon_ratio'] = df['lat'] / df['lon']
    df['depth_ratio'] = df['depth'] / (df['lat'] ** 2 + df['lon'] ** 2) ** 0.5
    
    # New features
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['min_sin'] = np.sin(2 * np.pi * df['min']/60)
    df['min_cos'] = np.cos(2 * np.pi * df['min']/60)
    df['sec_sin'] = np.sin(2 * np.pi * df['sec']/1000)
    df['sec_cos'] = np.cos(2 * np.pi * df['sec']/1000)
    
    # Interaction features
    df['depth_lat'] = df['depth'] * df['lat']
    df['depth_lon'] = df['depth'] * df['lon']
    df['lat_lon'] = df['lat'] * df['lon']
    
    # Time differences from reference point
    df['days_from_2014'] = (df['year'] - 2014) * 365 + df['month'] * 30 + df['day']
    df['minutes_in_day'] = df['hour'] * 60 + df['min']
    df['total_seconds'] = df['minutes_in_day'] * 60 + df['sec']/1000
    
    # Geographic zones (discretized features)
    df['lat_zone'] = pd.qcut(df['lat'], q=5, labels=False, duplicates='drop')
    df['lon_zone'] = pd.qcut(df['lon'], q=5, labels=False, duplicates='drop')
    df['depth_zone'] = pd.qcut(df['depth'], q=5, labels=False, duplicates='drop')
    
    # Additional geographic features
    df['distance_from_center'] = np.sqrt((df['lat'] - df['lat'].mean())**2 + 
                                       (df['lon'] - df['lon'].mean())**2)
    
    return df

# Add features to both train and test
train_data = add_features(train_data)
test_data = add_features(test_data)

# Separate features for different models
time_columns = ['year', 'month', 'day', 'hour', 'min', 'sec', 
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'day_sin', 'day_cos', 'min_sin', 'min_cos',
                'sec_sin', 'sec_cos', 'days_from_2014', 
                'minutes_in_day', 'total_seconds']

location_columns = ['lat', 'lon', 'depth', 'dist_from_equator', 
                   'lat_lon_ratio', 'depth_ratio', 'depth_lat',
                   'depth_lon', 'lat_lon', 'distance_from_center',
                   'lat_zone', 'lon_zone', 'depth_zone']

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
    def __init__(self, input_size=19, hidden_size=256, num_layers=4, dropout=0.3):
        super(LSTMPredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers//2, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, num_layers//2, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size*2, num_heads=4)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, 6)
        
    def forward(self, x):
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # Self-attention
        attn_out, _ = self.attention(lstm2_out.transpose(0,1), 
                                   lstm2_out.transpose(0,1), 
                                   lstm2_out.transpose(0,1))
        attn_out = attn_out.transpose(0,1)
        
        # Take last output and apply fully connected layers
        out = self.fc1(attn_out[:, -1, :])
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        predictions = self.fc3(out)
        return predictions

# Prepare data
sequence_length = 15  # Increased sequence length
batch_size = 64      # Increased batch size

# Scale the data
time_scaler = RobustScaler()  # Changed to RobustScaler for better handling of outliers
location_scaler = RobustScaler()
magnitude_scaler = RobustScaler()

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

# Initialize time_model
time_model = LSTMPredictor().to(device)

# Create base models with optimized parameters
base_models = [
    ('ridge', Ridge(alpha=0.1)),
    ('gbm', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, 
                                     max_depth=5, subsample=0.8)),
    ('rf', RandomForestRegressor(n_estimators=300, max_depth=12, 
                                min_samples_split=5, min_samples_leaf=2)),
    ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
]

# Create stacking models for each location feature
location_models = {
    'lat': StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=0.05),
        cv=5
    ),
    'lon': StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=0.05),
        cv=5
    ),
    'depth': StackingRegressor(
        estimators=base_models,
        final_estimator=GradientBoostingRegressor(n_estimators=200),
        cv=5
    )
}

# Optimized parameters for magnitude model
magnitude_params = {
    'n_estimators': 300,
    'max_depth': 12,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt'
}

magnitude_model = RandomForestRegressor(**magnitude_params)

# Train LSTM model
criterion = nn.L1Loss()  # Changed to L1Loss (MAE)
optimizer = torch.optim.Adam(time_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

best_val_loss = float('inf')
patience = 15  # Increased patience
patience_counter = 0
min_lr = 1e-6  # Minimum learning rate threshold

num_epochs = 200  # Increased epochs
for epoch in range(num_epochs):
    time_model.train()
    train_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = time_model(batch_x)
        loss = criterion(outputs, batch_y[:, :6])
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(time_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    # Calculate average training loss
    train_loss = train_loss / len(train_loader)
    
    # Validation
    time_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = time_model(batch_x)
            val_loss += criterion(outputs, batch_y[:, :6]).item()
    
    # Calculate average validation loss
    val_loss = val_loss / len(val_loader)
    
    # Step the scheduler with validation loss
    scheduler.step(val_loss)
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': time_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience or current_lr < min_lr:
        print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
        break
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')

# Load best model
checkpoint = torch.load('best_model.pth')
time_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['best_val_loss']:.4f}")

# Train location models
for feature, model in location_models.items():
    idx = location_columns.index(feature)
    model.fit(train_data[location_columns], train_data[location_columns[idx]])

# Train magnitude model with more features
magnitude_features = location_columns + ['year', 'month', 'day', 'hour', 'min', 'sec']
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
    test_sequences = test_sequences.to(device)
    time_predictions = time_model(test_sequences[-1].unsqueeze(0))
    time_predictions = time_predictions.cpu()
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

def apply_physical_constraints(submission):
    # Ensure time constraints
    submission['year_as'] = submission['year_as'].clip(2014, 2024)
    submission['month_as'] = submission['month_as'].clip(1, 12)
    submission['day_as'] = submission['day_as'].clip(1, 31)
    submission['hour_as'] = submission['hour_as'].clip(0, 23)
    submission['min_as'] = submission['min_as'].clip(0, 59)
    submission['sec_as'] = submission['sec_as'].clip(0, 999)
    
    # Ensure location constraints based on training data bounds
    lat_min, lat_max = train_data['lat'].min(), train_data['lat'].max()
    lon_min, lon_max = train_data['lon'].min(), train_data['lon'].max()
    depth_min, depth_max = train_data['depth'].min(), train_data['depth'].max()
    
    submission['lat_as'] = submission['lat_as'].clip(lat_min, lat_max)
    submission['lon_as'] = submission['lon_as'].clip(lon_min, lon_max)
    submission['depth_as'] = submission['depth_as'].clip(depth_min, depth_max)
    
    # Ensure magnitude constraints
    class_min, class_max = train_data['class'].min(), train_data['class'].max()
    submission['class_as'] = submission['class_as'].clip(class_min, class_max)
    
    return submission

# Apply constraints before saving
submission = apply_physical_constraints(submission)

# Save predictions
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv") 