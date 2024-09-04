import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import nfl_data_py as nfl
from get_players import get_players
from predict_passing_yards import predict_passing_yards

# Load and prepare data
def load_data():
    schedule = nfl.import_schedules([2024])
    schedule['gameday'] = pd.to_datetime(schedule['gameday'])
    current_date = datetime.now().date()
    end_date = current_date + timedelta(days=7)
    next_7_days_games = schedule[(schedule['gameday'].dt.date >= current_date) & 
                                 (schedule['gameday'].dt.date <= end_date)]
    return next_7_days_games.sort_values('gameday')

# Streamlit app
def main():
    st.title("NFL Passing Yards Predictor")

    # Load data
    games = load_data()

    # Game selection
    selected_game = st.selectbox("Select a game:", 
                                 [f"{game['gameday'].strftime('%Y-%m-%d %H:%M')}: {game['away_team']} @ {game['home_team']}" 
                                  for _, game in games.iterrows()])

    # Get selected game details
    game_index = games.index[games.apply(lambda row: f"{row['gameday'].strftime('%Y-%m-%d %H:%M')}: {row['away_team']} @ {row['home_team']}" == selected_game, axis=1)].tolist()[0]
    game = games.loc[game_index]

    # Get players for both teams
    home_qb, home_receivers = get_players(game['home_team'])
    away_qb, away_receivers = get_players(game['away_team'])

    # Team selection
    selected_team = st.radio("Select a team:", [game['home_team'], game['away_team']])

    # QB and receiver selection
    if selected_team == game['home_team']:
        qb = st.selectbox("Select QB:", [home_qb])
        receivers = home_receivers
        opponent_team = game['away_team']
    else:
        qb = st.selectbox("Select QB:", [away_qb])
        receivers = away_receivers
        opponent_team = game['home_team']

    # Set default index to 0, 1, and 2 for the top 3 targeted players
    wr1 = st.selectbox("Select Target 1:", receivers, index=0)
    wr2 = st.selectbox("Select Target 2:", receivers, index=min(1, len(receivers)-1))
    wr3 = st.selectbox("Select Target 3:", receivers, index=min(2, len(receivers)-1))

    # Predict button
    if st.button("Predict Passing Yards"):
        try:
            predicted_yards = predict_passing_yards(qb, wr1, wr2, wr3, opponent_team)
            st.success(f"Predicted passing yards for {qb}: {predicted_yards:.2f}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
