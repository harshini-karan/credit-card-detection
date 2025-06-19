from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, show_plots=True, model_name="Model"):
    """
    Print evaluation metrics and plot confusion matrix and ROC curve.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"{model_name} ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    if show_plots:
        plot_confusion_matrix(y_test, y_pred, title=f"{model_name} Confusion Matrix")
        plot_roc_curve(y_test, y_proba, title=f"{model_name} ROC Curve")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.4f})'.format(roc_auc_score(y_true, y_proba)))
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    # Example usage (for testing)
    from data_loader import load_data
    from preprocess import preprocess
    from model import train_random_forest

    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test, show_plots=True, model_name="Random Forest")
