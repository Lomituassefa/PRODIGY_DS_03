import sys
from pathlib import Path

# Resolve absolute path to PRODIGY_DS_03 repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prepend project root for src module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, roc_auc_score

from src.preprocessing import load_data


def plot_roc_curve():
    # Load test dataset splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Load trained model artifact
    model_path = PROJECT_ROOT / "models" / "best_decision_tree.joblib"
    best_model = joblib.load(model_path)

    # Predict class probabilities
    y_probs = best_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_probs)

    # Plot ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    display = RocCurveDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        name=f"Decision Tree (AUC = {auc_score:.3f})",
        ax=ax,
        plot_chance_level=True,
    )

    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save output plot
    images_dir = PROJECT_ROOT / "images"
    images_dir.mkdir(exist_ok=True)
    save_path = images_dir / "roc_curve.png"

    plt.savefig(save_path, dpi=300)
    print(f"Saved ROC curve plot to: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_roc_curve()