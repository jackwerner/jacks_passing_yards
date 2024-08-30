#load nfl_qb_prediction_dataset_v2.csv
import pandas as pd

df = pd.read_csv('nfl_qb_prediction_dataset_v2.csv')

print(df.head())

df = df.drop(['player_id', 'player_name', 'season', 'week', 'passing_yards', 'WR_1_player_id', 'WR_2_player_id', 'WR_3_player_id'], axis=1)

print(len(df.columns))
print(df.columns)