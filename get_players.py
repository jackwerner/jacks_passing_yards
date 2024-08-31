import nfl_data_py as nfl
import pandas as pd
from datetime import datetime, date

# Import schedule and weekly stats
schedule = nfl.import_schedules([2024])
weekly_stats = nfl.import_weekly_data([2023])

# Get the first game from the schedule
first_game = schedule.iloc[0]

# Get the home and away teams
home_team = first_game['home_team']
away_team = first_game['away_team']

# Function to get QB and top 3 receivers by average targets per game
def get_players(team):
    team_stats = weekly_stats[weekly_stats['recent_team'] == team]
    # Get QB
    qb_stats = team_stats[team_stats['position'] == 'QB']
    qb = qb_stats.groupby('player_display_name')['passing_yards'].sum().sort_values(ascending=False).index[0] if not qb_stats.empty else "No QB found"
    
    # Calculate average targets per game for receivers
    receiver_stats = team_stats[team_stats['position'].isin(['WR', 'TE', 'RB'])]
    avg_targets = receiver_stats.groupby('player_display_name')['targets'].agg(['sum', 'count'])
    avg_targets['avg_targets_per_game'] = avg_targets['sum'] / avg_targets['count']
    
    # Get top 3 receivers by average targets per game
    top_receivers = avg_targets.sort_values('avg_targets_per_game', ascending=False).head(3).index.tolist()
    
    return qb, top_receivers

# Get players for both teams
home_qb, home_receivers = get_players(home_team)
away_qb, away_receivers = get_players(away_team)

print(f"First game: {away_team} @ {home_team}")
print(f"{home_team} QB: {home_qb}")
print(f"{home_team} Top 3 Receivers: {', '.join(home_receivers)}")
print(f"{away_team} QB: {away_qb}")
print(f"{away_team} Top 3 Receivers: {', '.join(away_receivers)}")

