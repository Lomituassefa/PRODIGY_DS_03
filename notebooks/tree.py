import sys
from pathlib import Path

# Resolve absolute path to PRODIGY_DS_03 repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prepend project root for src module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from src.preprocessing import load_data


def plot_decision_tree():
    # Load feature matrix
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Extract feature names if available as a DataFrame, otherwise pass None
    feature_names = X_train.columns.tolist() if hasattr(X_train, "columns") else None

    # Load serialized Decision Tree model artifact
    model_path = PROJECT_ROOT / "models" / "best_decision_tree.joblib"
    best_model = joblib.load(model_path)

    # Render Decision Tree Structure (top 3 levels for readability)
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        best_model,
        max_depth=3,
        feature_names=feature_names,
        class_names=["No Subscription", "Subscription"],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )

    plt.title("Decision Tree Structure (Top Depths)", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save output plot
    images_dir = PROJECT_ROOT / "images"
    images_dir.mkdir(exist_ok=True)
    save_path = images_dir / "tree_structure.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved tree structure plot to: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_decision_tree()