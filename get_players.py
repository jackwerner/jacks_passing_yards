import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime, date

# Import schedule and depth charts
schedule = nfl.import_schedules([2024])
depth_charts = nfl.import_depth_charts([2023])

# Get the first game from the schedule
first_game = schedule.iloc[0]

# Get the home and away teams
home_team = first_game['home_team']
away_team = first_game['away_team']

# Function to get QB and top 3 WRs for a team
def get_players(team):
    team_depth = depth_charts[depth_charts['club_code'] == team]
    
    # Convert depth_team to numeric, replacing non-numeric values with NaN
    team_depth['depth_team'] = pd.to_numeric(team_depth['depth_team'], errors='coerce')
    
    # Calculate average depth position for each player, ignoring NaN values
    avg_depth = team_depth.groupby(['full_name', 'position'])['depth_team'].mean().reset_index()
    
    # Get QB
    qb = avg_depth[(avg_depth['position'] == 'QB') & avg_depth['depth_team'].notna()].sort_values('depth_team').iloc[0]['full_name'] if not avg_depth[(avg_depth['position'] == 'QB') & avg_depth['depth_team'].notna()].empty else "No QB found"
    
    # Get top 3 WRs
    wrs = avg_depth[(avg_depth['position'] == 'WR') & avg_depth['depth_team'].notna()].sort_values('depth_team').head(3)['full_name'].tolist()
    
    return qb, wrs

# Get players for both teams
home_qb, home_wrs = get_players(home_team)
away_qb, away_wrs = get_players(away_team)

print(f"First game: {away_team} @ {home_team}")
print(f"{home_team} QB: {home_qb}")
print(f"{home_team} Top 3 WRs: {', '.join(home_wrs)}")
print(f"{away_team} QB: {away_qb}")
print(f"{away_team} Top 3 WRs: {', '.join(away_wrs)}")

