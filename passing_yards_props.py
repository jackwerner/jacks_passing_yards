import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import nfl_data_py as nfl
from get_players import get_players
from predict_passing_yards import predict_passing_yards
import csv
from odds_api import get_qb_passing_yards_odds  # Updated import

st.set_page_config(layout="wide")

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

    st.subheader("Upcoming Games and Passing Yard Predictions")

    # Create a list to store all predictions
    all_predictions = []

    for _, game in games.iterrows():
        st.write(f"**{game['gameday'].strftime('%m/%d')} - {game['away_team']} @ {game['home_team']}**")

        # Get players for both teams
        home_qb, home_receivers = get_players(game['home_team'])
        away_qb, away_receivers = get_players(game['away_team'])

        # Create two rows for home and away teams
        for team, qb, receivers in [(game['home_team'], home_qb, home_receivers),
                                    (game['away_team'], away_qb, away_receivers)]:
            cols = st.columns([2, 3, 3, 3, 3, 3])  # Add one more column
            
            with cols[0]:
                st.write(f"**{team}**")
                st.write(f"QB: {qb}")

            with cols[1]:
                st.write("Target 1:")
                wr1 = st.selectbox(f"", receivers, key=f"{team}_wr1_{_}", index=0)

            with cols[2]:
                st.write("Target 2:")
                wr2 = st.selectbox(f"", receivers, key=f"{team}_wr2_{_}", index=min(1, len(receivers)-1))

            with cols[3]:
                st.write("Target 3:")
                wr3 = st.selectbox(f"", receivers, key=f"{team}_wr3_{_}", index=min(2, len(receivers)-1))

            with cols[4]:
                st.write("Prediction:")
                try:
                    yards = predict_passing_yards(qb, wr1, wr2, wr3, game['away_team'] if team == game['home_team'] else game['home_team'])
                    st.success(f"{yards:.2f} yards")
                    
                    # Add prediction to the list
                    prediction = {
                        'Date': game['gameday'].strftime('%m/%d'),
                        'Team': team,
                        'QB': qb,
                        'Target 1': wr1,
                        'Target 2': wr2,
                        'Target 3': wr3,
                        'Opponent': game['away_team'] if team == game['home_team'] else game['home_team'],
                        'Predicted Yards': f"{yards:.2f}"
                    }

                    # Add Odds API data
                    with cols[5]:
                        st.write("Odds API:")
                        odds_info = get_qb_passing_yards_odds(team, qb)
                        if odds_info:
                            st.info(f"{odds_info['line']} yards")
                            prediction['Odds API Yards'] = f"{odds_info['line']}"
                            prediction['Odds API Over'] = f"{odds_info['over']}"
                            prediction['Odds API Under'] = f"{odds_info['under']}"
                        else:
                            st.warning("N/A")
                            prediction['Odds API Yards'] = "N/A"
                            prediction['Odds API Over'] = "N/A"
                            prediction['Odds API Under'] = "N/A"

                    all_predictions.append(prediction)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.write("---")

    # Add a button to save predictions to CSV
    if st.button("Save Predictions to CSV"):
        df = pd.DataFrame(all_predictions)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="nfl_passing_yards_predictions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
