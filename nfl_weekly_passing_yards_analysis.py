import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt

# Get weekly stats for the last 5 seasons (2019-2023)
seasons = list(range(2019, 2024))
weekly_stats = nfl.import_weekly_data(seasons)

# Filter for quarterbacks' passing yards and group by week
passing_stats = weekly_stats[(weekly_stats['passing_yards'].notna()) & (weekly_stats['position'] == 'QB')]
avg_passing_yards = passing_stats.groupby('week')['passing_yards'].mean().reset_index()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(avg_passing_yards['week'], avg_passing_yards['passing_yards'], marker='o')
plt.title('Average Weekly Passing Yards for Quarterbacks (2019-2023)')
plt.xlabel('Week')
plt.ylabel('Average Passing Yards')
plt.grid(True)
plt.show()

# Print week 1 average and overall average for comparison
week_1_avg = avg_passing_yards.loc[avg_passing_yards['week'] == 1, 'passing_yards'].values[0]
overall_avg = avg_passing_yards['passing_yards'].mean()

print(f"Week 1 average passing yards for quarterbacks: {week_1_avg:.2f}")
print(f"Overall average passing yards for quarterbacks: {overall_avg:.2f}")