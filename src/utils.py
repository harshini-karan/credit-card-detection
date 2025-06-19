import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(y, title="Class Distribution"):
    """
    Plot the distribution of classes (fraud vs. non-fraud).
    """
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Class (0=Non-Fraud, 1=Fraud)")
    plt.ylabel("Count")
    plt.show()

def print_dataset_info(df):
    """
    Print basic info about the dataset.
    """
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())
    print("Class distribution:\n", df['Class'].value_counts())

def scale_single_transaction(transaction, scaler):
    """
    Scales a single transaction's 'Amount' field using the provided scaler.
    Expects transaction as a dict or pandas Series.
    """
    amount = np.array(transaction['Amount']).reshape(-1, 1)
    amount_scaled = scaler.transform(amount)
    transaction['Amount_scaled'] = amount_scaled[0][0]
    del transaction['Amount']
    return transaction

# You can add more utility functions as needed.

if __name__ == "__main__":
    import pandas as pd
    from data_loader import load_data
    df = load_data()
    print_dataset_info(df)
    plot_class_distribution(df['Class'])
