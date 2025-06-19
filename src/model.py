import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os

def train_random_forest(X_train, y_train, model_path='rf_model.joblib'):
    """
    Train a Random Forest classifier and save the model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Random Forest model saved to {model_path}")
    return model

def train_logistic_regression(X_train, y_train, model_path='lr_model.joblib'):
    """
    Train a Logistic Regression classifier and save the model.
    """
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Logistic Regression model saved to {model_path}")
    return model

def load_model(model_path):
    """
    Load a trained model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    return model

if __name__ == "__main__":
    # Example usage (for testing)
    from data_loader import load_data
    from preprocess import preprocess

    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    train_random_forest(X_train, y_train)
    train_logistic_regression(X_train, y_train)
