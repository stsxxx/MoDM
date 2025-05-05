import pandas as pd

# Load the CSV file
file_path = "/home/stilex/diffusion-opt/serving/request_throughput_N_1000_log2.csv"
df = pd.read_csv(file_path)

# Create groups of 5-minute intervals
df['group'] = (df.index // 7)  

# Compute the average for each group
aggregated_df = df.groupby('group', as_index=False).mean()

# Drop the 'group' column as it's no longer needed
aggregated_df.drop(columns=['group'], inplace=True)


print(aggregated_df)