import pandas as pd
import numpy as np
from urllib.parse import urljoin
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import json
import urllib.error
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

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

# Load and process defense stats
def load_defense_stats():
    defense_url = "pfr_advstats/advstats_season_def.csv"
    defense_stats = load_data(defense_url)
    defense_stats = defense_stats[defense_stats['season'] == 2023]  # Filter for 2023 season
    
    # Calculate defensive stats
    defense_stats = defense_stats.groupby('tm').agg({
        'int': 'mean',
        'tgt': 'mean',
        'cmp': 'mean',
        'yds': 'mean',
        'td': 'mean',
        'rat': 'mean',
        'dadot': 'mean',
        'air': 'mean',
        'yac': 'mean',
        'bltz': 'mean',
        'hrry': 'mean',
        'qbkd': 'mean',
        'sk': 'mean',
        'prss': 'mean',
        'comb': 'mean',
        'm_tkl': 'mean'
    }).reset_index()
    
    # Calculate additional stats
    defense_stats['def_cmp_percent'] = np.where(defense_stats['tgt'] != 0, defense_stats['cmp'] / defense_stats['tgt'], 0)
    defense_stats['def_yards_per_cmp'] = np.where(defense_stats['cmp'] != 0, defense_stats['yds'] / defense_stats['cmp'], 0)
    defense_stats['def_yards_per_tgt'] = np.where(defense_stats['tgt'] != 0, defense_stats['yds'] / defense_stats['tgt'], 0)
    defense_stats['def_missed_tackle_percent'] = np.where(defense_stats['comb'] != 0, defense_stats['m_tkl'] / defense_stats['comb'], 0)
    
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
    # Rename columns to add 'def_' prefix
    # defense_stats.columns = ['team' if col == 'tm' else f'def_{col}' for col in defense_stats.columns]
    
    return defense_stats

# Load defense stats
defense_stats = load_defense_stats()

def get_defense_stats(team):
    """Get the defensive stats for the opponent team."""
    team_stats = defense_stats[defense_stats['tm'] == team]
    if team_stats.empty:
        raise ValueError(f"No defensive stats found for team: {team}")
    return team_stats.iloc[0]

# Load feature names from the JSON file
with open('qb_passing_yards_model_features_v2.json', 'r') as f:
    expected_features = json.load(f)

print("Feature names loaded from 'qb_passing_yards_model_features_v2.json'")

# Load the trained model
model = xgb.XGBRegressor()
try:
    model.load_model('qb_passing_yards_model_v2.json')
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Please ensure the model file 'qb_passing_yards_model_v2.json' exists and is valid.")
    exit(1)

# Load weekly QB passing data (2023)
print("Loading QB data...")
qb_data = load_data("player_stats/player_stats_2023.csv")
qb_data = qb_data[qb_data['position'] == 'QB']

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

# Merge qb_data with season_pass
qb_data = qb_data.merge(season_pass, on=['player_id', 'season'], how='left')
print(f"QB data loaded and merged with season stats. Shape: {qb_data.shape}")

# Load receiving stats for 2023
rec_stats = load_data("pfr_advstats/advstats_season_rec.csv")
rec_stats = rec_stats[rec_stats['season'] == 2023]
rec_stats = rec_stats.rename(columns={'pfr_id': 'player_id', 'tm': 'team'})

# Calculate season-long average receiving stats
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
# Load the NFL IDs mapping file
nfl_ids = pd.read_csv('nfl_ids.csv')

# Merge qb_data with nfl_ids
qb_data = qb_data.merge(nfl_ids[['rotowire_id', 'gsis_id']], left_on='player_id', right_on='gsis_id', how='left')
qb_data = qb_data.drop('gsis_id', axis=1)

# merge rec_stats with nfl_ids
rec_stats = rec_stats.merge(nfl_ids[['rotowire_id', 'pfr_id']], left_on='player_id', right_on='pfr_id', how='left')
rec_stats = rec_stats.drop(['pfr_id', 'player_id'], axis=1)
rec_stats.rename(columns={'rotowire_id': 'player_id'}, inplace=True)

def get_player_id(name):
    """Get the rotowire_id from the NFL IDs mapping file."""
    player = nfl_ids[nfl_ids['name'] == name].iloc[0]
    return player['rotowire_id']

def get_player_stats(rotowire_id, position):
    """Get the player's average stats from the 2023 dataset."""
    if position == 'QB':
        # Use only the columns specified for QBs in the JSON file
        qb_columns = expected_features['qb_stats']
        player_data = qb_data[qb_data['rotowire_id'] == rotowire_id][qb_columns].mean()
    else:
        # Use only the columns specified for receivers in the JSON file
        wr_columns = expected_features['wr_stats']
        player_data = rec_stats[rec_stats['player_id'] == rotowire_id][wr_columns].mean()
    
    # Fill NaN values with 0
    player_data = player_data.fillna(0)
    
    return player_data

def predict_passing_yards(qb_name, wr1_name, wr2_name, wr3_name, opponent_team):
    # Get player IDs
    qb_id = get_player_id(qb_name)
    wr1_id = get_player_id(wr1_name)
    wr2_id = get_player_id(wr2_name)
    wr3_id = get_player_id(wr3_name)

    # Get player stats
    qb_stats = get_player_stats(qb_id, 'QB')
    wr1_stats = get_player_stats(wr1_id, 'WR')
    wr2_stats = get_player_stats(wr2_id, 'WR')
    wr3_stats = get_player_stats(wr3_id, 'WR')
    defense_stats = get_defense_stats(opponent_team)

    # Prepare input data
    input_data = pd.DataFrame({
        **qb_stats.to_dict(),
        **{f'WR_1_{k}': v for k, v in wr1_stats.items()},
        **{f'WR_2_{k}': v for k, v in wr2_stats.items()},
        **{f'WR_3_{k}': v for k, v in wr3_stats.items()},
        **defense_stats.to_dict()  # Include all defense stats directly
    }, index=[0])
    # Ensure all expected features are present
    for feature in expected_features['all']:
        if feature not in input_data.columns:
            print(f"Feature {feature} not found in input data.")
            input_data[feature] = 0

    # Reorder columns to match the expected feature order
    input_data = input_data[expected_features['all']]

    # Print defensive stats and final input data for debugging
    # print("Defensive stats for", opponent_team, ":", defense_stats.to_dict())
    # print("Final input data:")
    # print(input_data)

    # Make prediction
    prediction = model.predict(input_data)

    return prediction[0]

# Example usage
qb_name = "Daniel Jones"
wr1_name = "A.J. Brown"
wr2_name = "DeVonta Smith"
wr3_name = "Saquon Barkley"
opponent_team = "KC"

predicted_yards = predict_passing_yards(qb_name, wr1_name, wr2_name, wr3_name, opponent_team)
print(f"Predicted passing yards for {qb_name}: {predicted_yards:.2f}")