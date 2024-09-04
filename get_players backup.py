import nfl_data_py as nfl
import pandas as pd
from datetime import datetime, timedelta

# Import schedule
schedule = nfl.import_schedules([2024])

# Convert 'gameday' column to datetime
schedule['gameday'] = pd.to_datetime(schedule['gameday'])

# Get current date and date 7 days from now
current_date = datetime.now().date()
end_date = current_date + timedelta(days=7)

# Filter schedule for games in the next 7 days
next_7_days_games = schedule[(schedule['gameday'].dt.date >= current_date) & 
                             (schedule['gameday'].dt.date <= end_date)]

# Sort games by date and time
next_7_days_games = next_7_days_games.sort_values('gameday')

# Import weekly stats
weekly_stats = nfl.import_weekly_data([2023]) #need to change this to 2024

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

# Print games and player details for the next 7 days
print(f"Games and players in the next 7 days (from {current_date} to {end_date}):")
for _, game in next_7_days_games.iterrows():
    print(f"\n{game['gameday'].strftime('%Y-%m-%d %H:%M')}: {game['away_team']} @ {game['home_team']}")
    
    home_qb, home_receivers = get_players(game['home_team'])
    away_qb, away_receivers = get_players(game['away_team'])
    
    print(f"{game['home_team']} QB: {home_qb}")
    print(f"{game['home_team']} Top 3 Receivers: {', '.join(home_receivers)}")
    print(f"{game['away_team']} QB: {away_qb}")
    print(f"{game['away_team']} Top 3 Receivers: {', '.join(away_receivers)}")

