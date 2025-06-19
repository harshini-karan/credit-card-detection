# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using Python and scikit-learn.

## Features

- Data exploration and preprocessing
- Model training (Random Forest, Logistic Regression)
- Evaluation metrics (accuracy, precision, recall, F1)
- REST API for real-time prediction (FastAPI)

## Usage

1. Place `creditcard.csv` (from Kaggle) in the `data/` folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the pipeline: `python main.py`
4. Start the API: `uvicorn api.app:app --reload`
