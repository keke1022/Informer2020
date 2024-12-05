import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse

def clean_data(input_path, output_path):
    # Step 1: Load the dataset
    print("Loading data...")
    data = pd.read_csv(input_path)
    
    # Step 2: Convert the date column to datetime
    print("Converting date column...")
    data['date'] = pd.to_datetime(data['date'])
    
    # Step 3: Handle missing values (fill with mean)
    print("Handling missing values...")
    if data.isnull().values.any():
        data.fillna(data.mean(), inplace=True)
    
    # Step 4: Normalize numerical columns (zero-mean normalization)
    print("Normalizing numerical columns...")
    numerical_cols = data.columns.difference(['date'])  # Exclude the date column
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # Step 5: Save the cleaned data to a new file
    print(f"Saving cleaned data to {output_path}...")
    data.to_csv(output_path, index=False)
    print("Data cleaning completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and preprocess time series data")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output', type=str, required=True, help="Path to save the cleaned CSV file")
    args = parser.parse_args()
    
    clean_data(args.input, args.output)
