from src.data_loader import load_data
from src.preprocess import preprocess
from src.model import train_random_forest, train_logistic_regression
from src.evaluate import evaluate_model

def main():
    print("=== Credit Card Fraud Detection Pipeline ===")

    # 1. Load data
    print("Loading data...")
    df = load_data('data/creditcard.csv')

    # 2. Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # 3. Train models
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)

    # 4. Evaluate models
    print("\n--- Random Forest Evaluation ---")
    evaluate_model(rf_model, X_test, y_test, show_plots=True, model_name="Random Forest")

    print("\n--- Logistic Regression Evaluation ---")
    evaluate_model(lr_model, X_test, y_test, show_plots=True, model_name="Logistic Regression")

    print("Pipeline completed.")

if __name__ == "__main__":
    main()
