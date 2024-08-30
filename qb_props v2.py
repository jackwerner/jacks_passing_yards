import pandas as pd
import numpy as np
from urllib.parse import urljoin
import urllib.error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base URL for the data files
base_url = "https://github.com/nflverse/nflverse-data/releases/download/"

# Function to load data from URL
def load_data(url):
    try:
        return pd.read_csv(urljoin(base_url, url))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        print(f"Failed URL: {urljoin(base_url, url)}")
        raise

# Load weekly QB passing data (2018-2023)
print("Loading QB data...")
qb_data = pd.concat([load_data(f"player_stats/player_stats_{year}.csv") for year in range(2018, 2024)])
qb_data = qb_data[qb_data['position'] == 'QB']
# Filter and select only the specified columns

print(f"QB data loaded. Shape: {qb_data.shape}")

# Create season pass dataset
print("Creating season pass dataset...")
season_pass = qb_data.groupby(['player_id', 'season']).agg({
    'completions': ['count', 'mean', 'std', lambda x: x.sum() / qb_data.loc[x.index, 'attempts'].sum()],
    'attempts': ['mean', 'std'],
    'sacks': ['mean', 'std'],
    'sack_yards': 'mean',
    'sack_fumbles': 'mean',
    'passing_air_yards': ['mean', 'std', lambda x: x.sum() / qb_data.loc[x.index, 'attempts'].sum()],
    'passing_yards_after_catch': ['mean', 'std', lambda x: x.sum() / qb_data.loc[x.index, 'attempts'].sum()],
    'passing_first_downs': 'mean',
    'passing_epa': ['mean', 'std'],
    'passing_2pt_conversions': 'mean',
    'pacr': ['mean', 'var'],
    'dakota': ['mean', 'var'],
    'carries': ['mean', 'std'],
    'rushing_yards': ['mean', 'std']
}).reset_index()

# Flatten column names
season_pass.columns = ['player_id', 'season', 'games', 'completions_per_game', 'sd_completions_per_game', 'completions_per_attempt',
                       'attempts_per_game', 'sd_attempts_per_game', 'sacks_per_game', 'sd_sacks_per_game',
                       'sack_yards_per_game', 'sack_fumbles_per_game', 'passing_air_yards_per_game', 'sd_passing_air_yards_per_game',
                       'passing_air_yards_per_attempt', 'passing_yards_after_catch_per_game', 'sd_passing_yards_after_catch_per_game',
                       'passing_yards_after_catch_per_attempt', 'passing_first_downs_per_game', 'passing_epa_per_game',
                       'sd_passing_epa_per_game', 'passing_2pt_conversions_per_game', 'avg_pacr', 'var_pacr',
                       'avg_dakota', 'var_dakota', 'carries_per_game', 'sd_carries_per_game',
                       'rushing_yards_per_game', 'sd_rushing_yards_per_game']

print(f"Season pass dataset created. Shape: {season_pass.shape}")

# Load receiving stats
print("Loading receiving stats...")
rec_stats = load_data("pfr_advstats/advstats_season_rec.csv")
rec_stats = rec_stats.rename(columns={'pfr_id': 'player_id', 'tm': 'team'})
print(f"Receiving stats loaded. Shape: {rec_stats.shape}")

# Calculate season-long average receiving stats
print("Calculating season-long average receiving stats...")
rec_stats['season_rec_targets_p_game'] = rec_stats['tgt'] / rec_stats['g']
rec_stats['season_rec_p_game'] = rec_stats['rec'] / rec_stats['g']
rec_stats['season_rec_p_target'] = rec_stats['rec'] / rec_stats['tgt']
rec_stats['season_rec_yards_p_game'] = rec_stats['yds'] / rec_stats['g']
rec_stats['season_ybc_p_r'] = rec_stats['ybc_r']
rec_stats['season_yac_p_r'] = rec_stats['yac_r']
rec_stats['season_broken_tackle_p_game'] = rec_stats['brk_tkl'] / rec_stats['g']
rec_stats['season_drop_percent'] = rec_stats['drop_percent']
rec_stats['season_int_p_target'] = rec_stats['int'] / rec_stats['tgt']
rec_stats['season_adot'] = rec_stats['adot']

# Load target stats from the new data source
print("Loading target stats...")
target_stats = []
for year in range(2018, 2024):
    url = f"https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_{year}_receiving.csv.gz"
    df = load_data(url)
    df['season'] = year
    df = df[df['week'] != 0]  # Drop week zero
    target_stats.append(df)
target_stats = pd.concat(target_stats)
print(f"Receiving stats loaded. Shape: {target_stats.shape}")

# Load the NFL IDs mapping file
print("Loading NFL IDs mapping...")
nfl_ids = pd.read_csv('nfl_ids.csv')
print(f"NFL IDs mapping loaded. Shape: {nfl_ids.shape}")

# Merge target_stats with NFL IDs to get rotowire_id
target_stats = target_stats.merge(nfl_ids[['gsis_id', 'rotowire_id']], 
                                  left_on='player_gsis_id', 
                                  right_on='gsis_id', 
                                  how='left')
target_stats['player_id'] = target_stats['rotowire_id']#.fillna(target_stats['player_gsis_id'])
target_stats = target_stats.drop(columns=['gsis_id', 'rotowire_id', 'player_gsis_id'])

# Ensure rec_stats uses rotowire_id as player_id
rec_stats = rec_stats.merge(nfl_ids[['pfr_id', 'rotowire_id']], 
                            left_on='player_id', 
                            right_on='pfr_id', 
                            how='left')
rec_stats['player_id'] = rec_stats['rotowire_id']#.fillna(rec_stats['player_id'])
rec_stats = rec_stats.drop(columns=['pfr_id', 'rotowire_id'])

# Now merge target_stats with rec_stats using the correct player_id
print("Merging target_stats with rec_stats...")
target_stats = target_stats.rename(columns={'team_abbr': 'team'})
receiver_data = target_stats.merge(rec_stats, on=['player_id', 'season', 'team'], how='left')
print(f"Merged receiver data shape: {receiver_data.shape}")

# Get top 3 receivers by offense_pct for each team, season, and week
print("Identifying top 3 receivers for each game...")
top_receivers = receiver_data.sort_values('targets', ascending=False).groupby(['season', 'week', 'team']).head(3)

# Merge top_receivers with NFL IDs mapping
print("Merging top receivers with NFL IDs mapping...")
logger.info(f"top_receivers['player_id'] dtype: {top_receivers['player_id'].dtype}")
logger.info(f"nfl_ids['gsis_id'] dtype: {nfl_ids['gsis_id'].dtype}")

# Before merging, convert player_id to string in top_receivers
top_receivers['player_id'] = top_receivers['player_id'].astype(str)

# Ensure nfl_ids['gsis_id'] is also string type
nfl_ids['gsis_id'] = nfl_ids['gsis_id'].astype(str)

try:
    top_receivers = top_receivers.merge(nfl_ids[['gsis_id', 'rotowire_id']], 
                                        left_on='player_id', 
                                        right_on='gsis_id', 
                                        how='left')
    top_receivers['player_id'] = top_receivers['rotowire_id'].fillna(top_receivers['player_id'])
    top_receivers = top_receivers.drop(columns=['gsis_id', 'rotowire_id'])
except Exception as e:
    logger.error(f"Error during merge operation: {str(e)}")
    raise

# Create columns for each of the top 3 receivers' stats
receiver_stats = ['season_rec_targets_p_game', 'season_rec_p_game', 'season_rec_p_target', 'season_rec_yards_p_game', 
                  'season_ybc_p_r', 'season_yac_p_r', 'season_broken_tackle_p_game', 'season_drop_percent', 
                  'season_int_p_target', 'season_adot']

def safe_access(x, i):
    return x.iloc[i] if i < len(x) else pd.NA

# Create a new DataFrame to store only the desired columns
new_top_receivers = pd.DataFrame()
new_top_receivers['season'] = top_receivers['season']
new_top_receivers['week'] = top_receivers['week']
new_top_receivers['team'] = top_receivers['team']

for i in range(1, 4):
    new_top_receivers[f'WR_{i}_player_id'] = top_receivers.groupby(['season', 'week', 'team'])['player_id'].transform(lambda x: safe_access(x, i-1))
    for stat in receiver_stats:
        new_top_receivers[f'WR_{i}_{stat}'] = top_receivers.groupby(['season', 'week', 'team'])[stat].transform(lambda x: safe_access(x, i-1))

# Replace the original top_receivers DataFrame with the new one
top_receivers = new_top_receivers

# Handle missing values for receiver stats
for col in top_receivers.columns:
    if col.startswith('WR_'):
        if col.endswith('player_id'):
            top_receivers[col] = top_receivers[col].fillna('Unknown')
        else:
            top_receivers[col] = top_receivers[col].fillna(top_receivers[col].mean())

# Keep only one row per team, season, and week
top_receivers = top_receivers.groupby(['season', 'week', 'team']).first().reset_index()

# Merge top_receivers with the rest of the data
print("Merging top receivers data with QB data...")
qb_data = qb_data.rename(columns={'recent_team': 'team'})
final_data = qb_data.merge(top_receivers, on=['season', 'week', 'team'], how='left')
print(f"Final data shape after merging receivers: {final_data.shape}")

# Update the URL and season range
defense_url = "https://github.com/nflverse/nflverse-data/releases/download/pfr_advstats/advstats_season_def.csv"
defense_stats = pd.read_csv(defense_url)
defense_stats = defense_stats[defense_stats['season'].between(2018, 2023)]

print(f"Defense stats loaded. Shape: {defense_stats.shape}")
print(f"Defense stats columns: {defense_stats.columns.tolist()}")

# Rename 'tm' to 'team' in defense_stats for consistency
defense_stats = defense_stats.rename(columns={'tm': 'team'})

# Calculate defensive stats
print("Calculating defensive stats...")
defense_stats = defense_stats.groupby(['season', 'team']).agg({
    'int': lambda x: x.sum() / x.count(),
    'tgt': lambda x: x.sum() / x.count(),
    'cmp': lambda x: x.sum() / x.count(),
    'yds': lambda x: x.sum() / x.count(),
    'td': lambda x: x.sum() / x.count(),
    'rat': lambda x: (x * defense_stats.loc[x.index, 'gs']).sum() / defense_stats.loc[x.index, 'gs'].sum(),
    'dadot': lambda x: (x * defense_stats.loc[x.index, 'tgt']).sum() / defense_stats.loc[x.index, 'tgt'].sum(),
    'air': lambda x: x.sum() / x.count(),
    'yac': lambda x: x.sum() / x.count(),
    'bltz': lambda x: x.sum() / x.count(),
    'hrry': lambda x: x.sum() / x.count(),
    'qbkd': lambda x: x.sum() / x.count(),
    'sk': lambda x: x.sum() / x.count(),
    'prss': lambda x: x.sum() / x.count(),
    'comb': lambda x: x.sum() / x.count(),
    'm_tkl': lambda x: x.sum() / x.count(),
}).reset_index()

# Calculate additional stats
defense_stats['def_cmp_percent'] = defense_stats['cmp'] / defense_stats['tgt']
defense_stats['def_yards_per_cmp'] = defense_stats['yds'] / defense_stats['cmp']
defense_stats['def_yards_per_tgt'] = defense_stats['yds'] / defense_stats['tgt']
defense_stats['def_missed_tackle_percent'] = defense_stats['m_tkl'] / defense_stats['comb']

# Rename columns to match your R script and add 'def_' prefix
defense_stats = defense_stats.rename(columns={
    'int': 'def_interceptions',
    'tgt': 'def_targets',
    'cmp': 'def_completions',
    'yds': 'def_yards',
    'td': 'def_td',
    'rat': 'def_rating',
    'dadot': 'def_dadot',
    'air': 'def_air_yards',
    'yac': 'def_yards_after_catch',
    'bltz': 'def_blitz',
    'hrry': 'def_hurry',
    'qbkd': 'def_qbkd',
    'sk': 'def_sacks',
    'prss': 'def_pressures',
    'comb': 'def_combined_tackles',
    'm_tkl': 'def_missed_tackles'
})

print(f"Defensive stats calculated. Shape: {defense_stats.shape}")
print(f"Defensive stats columns: {defense_stats.columns.tolist()}")

# Merge all data
print("Merging all data...")
print(f"QB data shape: {qb_data.shape}")
print(f"Season pass shape: {season_pass.shape}")
final_data = final_data.merge(season_pass, on=['player_id', 'season'], suffixes=('', '_season'))
print(f"After first merge shape: {final_data.shape}")

# Check for missing values in receiver columns
receiver_columns = [col for col in final_data.columns if col.startswith('WR_')]
missing_receiver_data = final_data[receiver_columns].isnull().sum()
print("Missing values in receiver columns:")
print(missing_receiver_data)

# Handle missing values for receiver stats
for col in receiver_columns:
    if col.endswith('player_id'):
        final_data[col] = final_data[col].fillna('Unknown')
    else:
        final_data[col] = final_data[col].fillna(final_data[col].mean())

print(f"Top receivers shape: {top_receivers.shape}")
print(f"Defense stats shape: {defense_stats.shape}")
final_data = final_data.merge(defense_stats, left_on=['season', 'opponent_team'], right_on=['season', 'team'], suffixes=('', '_defense'))
print(f"After final merge shape: {final_data.shape}")

# Remove duplicate columns and rename as needed
columns_to_drop = [col for col in final_data.columns if col.endswith('_x') or col.endswith('_y')]
final_data = final_data.drop(columns=columns_to_drop)

# Rename any remaining problematic columns
column_rename_map = {
    'team_receiver': 'team',
    'team_defense': 'opponent_team'
}
final_data = final_data.rename(columns=column_rename_map)

print(f"Final data columns: {final_data.columns.tolist()}")

# Adjust the columns_to_keep list
columns_to_keep = [
    'player_id', 'player_name', 'season', 'week', 'completions', 'attempts', 'passing_yards', 'passing_tds',
    'interceptions', 'completions_per_game', 'sd_completions_per_game', 'completions_per_attempt',
    'attempts_per_game', 'sd_attempts_per_game', 'sacks_per_game', 'sd_sacks_per_game',
    'sack_yards_per_game', 'sack_fumbles_per_game', 'passing_air_yards_per_game', 'sd_passing_air_yards_per_game',
    'passing_air_yards_per_attempt', 'passing_yards_after_catch_per_game', 'sd_passing_yards_after_catch_per_game',
    'passing_yards_after_catch_per_attempt', 'passing_first_downs_per_game', 'passing_epa_per_game',
    'sd_passing_epa_per_game', 'passing_2pt_conversions_per_game', 'avg_pacr', 'var_pacr',
    'avg_dakota', 'var_dakota', 'carries_per_game', 'sd_carries_per_game',
    'rushing_yards_per_game', 'sd_rushing_yards_per_game', 'WR_1_player_id', 'WR_1_season_rec_targets_p_game',
    'WR_1_season_rec_p_game', 'WR_1_season_rec_p_target', 'WR_1_season_rec_yards_p_game', 'WR_1_season_ybc_p_r',
    'WR_1_season_yac_p_r', 'WR_1_season_broken_tackle_p_game', 'WR_1_season_drop_percent', 'WR_1_season_int_p_target',
    'WR_1_season_adot', 'WR_2_player_id', 'WR_2_season_rec_targets_p_game', 'WR_2_season_rec_p_game',
    'WR_2_season_rec_p_target', 'WR_2_season_rec_yards_p_game', 'WR_2_season_ybc_p_r', 'WR_2_season_yac_p_r',
    'WR_2_season_broken_tackle_p_game', 'WR_2_season_drop_percent', 'WR_2_season_int_p_target', 'WR_2_season_adot',
    'WR_3_player_id', 'WR_3_season_rec_targets_p_game', 'WR_3_season_rec_p_game', 'WR_3_season_rec_p_target',
    'WR_3_season_rec_yards_p_game', 'WR_3_season_ybc_p_r', 'WR_3_season_yac_p_r', 'WR_3_season_broken_tackle_p_game',
    'WR_3_season_drop_percent', 'WR_3_season_int_p_target', 'WR_3_season_adot', 'def_interceptions', 'def_targets', 'def_completions', 'def_yards', 'def_td',
    'def_rating', 'def_dadot', 'def_air_yards', 'def_yards_after_catch',
    'def_blitz', 'def_hurry', 'def_qbkd', 'def_sacks', 'def_pressures',
    'def_combined_tackles', 'def_missed_tackles', 'def_cmp_percent',
    'def_yards_per_cmp', 'def_yards_per_tgt', 'def_missed_tackle_percent'
]
# Check if all columns in columns_to_keep exist in final_data
missing_columns = [col for col in columns_to_keep if col not in final_data.columns]
if missing_columns:
    print(f"Warning: The following columns are missing from the final dataset: {missing_columns}")
    columns_to_keep = [col for col in columns_to_keep if col in final_data.columns]

final_dataset = final_data[columns_to_keep]
# Before saving the final dataset
print("Final dataset sample:")
print(final_dataset[['player_id', 'player_name', 'season', 'week', 'WR_1_player_id', 'WR_1_season_rec_targets_p_game']].head())

# Save the final dataset
final_dataset.to_csv('nfl_qb_prediction_dataset_v2.csv', index=False)

print("Dataset creation completed. Saved as 'nfl_qb_prediction_dataset_v2.csv'")

# Add XGBoost model with GridSearch
print("Training XGBoost model with GridSearch...")

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Prepare the data
X = final_dataset.drop(['player_id', 'player_name', 'season', 'week', 'passing_yards', 'WR_1_player_id', 'WR_2_player_id', 'WR_3_player_id'], axis=1)
y = final_dataset['passing_yards']

# Handle missing values
X = X.fillna(X.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Create the XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Perform GridSearch
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = best_model.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 Most Important Features:")
for feature, importance in sorted_features[:10]:
    print(f"{feature}: {importance}")

# Save the model
best_model.save_model('qb_passing_yards_model_v2.json')
print("Model saved as 'qb_passing_yards_model_v2.json'")


