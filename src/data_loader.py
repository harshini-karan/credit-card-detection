import pandas as pd
import os

def load_data(path='data/creditcard.csv'):
    """
    Loads the credit card fraud dataset from the specified path.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please place 'creditcard.csv' in the 'data/' folder.")
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    # Test loading
    df = load_data()
    print(df.head())
    print(f"Dataset shape: {df.shape}")
