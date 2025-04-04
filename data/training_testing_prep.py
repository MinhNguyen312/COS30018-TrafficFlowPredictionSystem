import os
import pandas as pd

def split_and_save_data(location_number):
    """Split data for a single location into train/test sets and save"""
    input_file = f'{location_number}_flow_processed.csv'
    
    # Skip if input file doesn't exist
    if not os.path.exists(input_file):
        print(f"Skipping {location_number} - file not found")
        return
        
    # Create train and test directories if they don't exist
    train_dir = 'train_data'
    test_dir = 'test_data'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Load and split data
    df = pd.read_csv(input_file)
    split_idx = int(0.7 * len(df))
    
    train = df[:split_idx]
    test = df[split_idx:]
    
    # Save split data
    train.to_csv(f'{train_dir}/{location_number}_flow.csv', index=False)
    test.to_csv(f'{test_dir}/{location_number}_flow.csv', index=False)
    print(f"Processed {location_number}: {len(train)} train, {len(test)} test samples")

def main():
    # Get list of all processed data files
    # files = [f for f in os.listdir('.') if f.endswith('_flow_processed.csv')]
    # location_numbers = [f.split('_')[0] for f in files]
    
    # print(f"Found {len(location_numbers)} locations to process")
    
    # # Process each location
    # for location in location_numbers:
        split_and_save_data(970)
        
if __name__ == "__main__":
    main()