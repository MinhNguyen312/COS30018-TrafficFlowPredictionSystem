import pandas as pd

# Load CSV
df = pd.read_csv('2000_flow_processed.csv')

split_idx = int(0.7 * len(df))

train = df[:split_idx]
test = df[split_idx:]


train.to_csv('2000_training_data/2000_flow_train.csv', index=False)
test.to_csv('2000_training_data/2000_flow_test.csv', index=False)