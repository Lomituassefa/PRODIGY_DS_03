import sys
from pathlib import Path

# Append parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tabulate import tabulate
from src.preprocessing import load_data
def evaluate_model():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    model_path = (
        Path(__file__).resolve().parents[1]/ "models"/ "best_decision_tree.joblib"
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please run src/train.py first."
        )
    best_model = joblib.load(model_path)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:,1]
    # core metrics
    test_auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # tabulate
    metrics_summary = [
        [
            "Optimal Decision Tree",
            f"{test_auc:.4f}",
            f"{acc:.4f}",
            f"{prec:.4f}",
            f"{rec:.4f}",
            f"{f1:.4f}",
        ]
    ]

    headers = [
        "Model Setup",
        "Test ROC-AUC",
        "Accuracy",
        "Precision (1)",
        "Recall (1)",
        "F1-Score (1)",
    ]
    print("\n" + "=" * 70)
    print("           FINAL UNSEEN TEST SET PERFORMANCE")
    print("=" * 70)
    print(tabulate(metrics_summary, headers=headers, tablefmt="fancy_grid"))
    # Print detailed classification report
    print("\n--- DETAILED TEST CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, digits=4))
    # Print raw confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_table = [
        ["Actual Negative (0)", f"TN: {tn}", f"FP: {fp}"],
        ["Actual Positive (1)", f"FN: {fn}", f"TP: {tp}"],
    ]
    cm_headers = ["", "Predicted Negative (0)", "Predicted Positive (1)"]
    print("\n--- TEST SET CONFUSION MATRIX ---")
    print(tabulate(cm_table, headers=cm_headers, tablefmt="fancy_grid"))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_model()