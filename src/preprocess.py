import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess(df, scaler_path='scaler.joblib', save_scaler=True):
    """
    Preprocess the credit card fraud dataset:
    - Drops 'Time' column (if present)
    - Scales 'Amount' and appends as 'Amount_scaled'
    - Drops original 'Amount'
    - Splits into train and test sets
    - Saves the scaler for later use

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Drop 'Time' if present
    if 'Time' in df.columns:
        df = df.drop('Time', axis=1)

    # Scale 'Amount'
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df = df.drop('Amount', axis=1)

    # Save scaler if needed
    if save_scaler:
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    # Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train-test split (stratify to preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler

def load_scaler(scaler_path='scaler.joblib'):
    """
    Load a saved scaler from disk.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    return scaler

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    print("Preprocessing complete. Shapes:")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
