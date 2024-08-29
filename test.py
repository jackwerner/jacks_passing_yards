import pandas as pd

# Read the CSV file
df = pd.read_csv('nfl_ids.csv')

# Count NA's in each column
na_counts = df.isna().sum()

# Print the results
for column, count in na_counts.items():
    print(f"{column}: {count}")
