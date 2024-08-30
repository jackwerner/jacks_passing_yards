# evaluate the model qb_passing_yards_model_features_v2.json
# the data is nfl_qb_prediction_dataset_v2.csv


import json
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the model
model = xgb.XGBRegressor()
model.load_model('qb_passing_yards_model_v2.json')

# Load the data
data = pd.read_csv('nfl_qb_prediction_dataset_v2.csv')

# Prepare the data (ensure this matches your training feature set)
X = data.drop(['player_id', 'player_name', 'season', 'week', 'passing_yards', 'WR_1_player_id', 'WR_2_player_id', 'WR_3_player_id'], axis=1)
y = data['passing_yards']

# Handle missing values
X = X.fillna(X.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Make predictions on test set
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Passing Yards')
plt.ylabel('Predicted Passing Yards')
plt.title('Actual vs Predicted Passing Yards (Test Set)')

# Add text with metrics
plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nR-squared: {r2:.2f}', transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig('actual_vs_predicted_passing_yards_test.png')
plt.close()

# Print metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Function to make predictions for new data
def predict_passing_yards(new_data):
    # Ensure new_data has the same columns as X
    new_data = new_data[X.columns]
    
    # Handle missing values
    new_data = new_data.fillna(X.mean())
    
    # Scale the features
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions
    predictions = model.predict(new_data_scaled)
    
    return predictions

# Example usage:
# new_player_data = pd.DataFrame(...)  # Create a DataFrame with the same features as X
# predicted_yards = predict_passing_yards(new_player_data)
# print(f"Predicted passing yards: {predicted_yards[0]:.2f}")
