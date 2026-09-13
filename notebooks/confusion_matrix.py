import sys
from pathlib import Path

# Resolve absolute path to repo root (PRODIGY_DS_03)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Append repository root for src module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.preprocessing import load_data


def confusion_matr_plot():
    # Load dataset splits
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Model path
    model_path = PROJECT_ROOT / "models" / "best_decision_tree.joblib"
    best_model = joblib.load(model_path)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        display_labels=["No Subscription", "Subscription"],
        cmap="Blues",
        ax=ax,
        colorbar=True,
    )
    plt.title("Test Set Confusion Matrix", fontsize=14, fontweight="bold")
    plt.grid(False)
    plt.tight_layout()

    # Save output
    images_dir = PROJECT_ROOT / "images"
    images_dir.mkdir(exist_ok=True)
    save_path = images_dir / "confusion_matrix.png"

    plt.savefig(save_path, dpi=300)
    print(f"Saved confusion matrix plot to: {save_path}")
    plt.show()
if __name__ == "__main__":
    confusion_matr_plot()