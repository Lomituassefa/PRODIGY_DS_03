import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from src.preprocessing import load_data
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV

def baseline():
    X_train, X_val, X_test, y_train, y_val, y_test=load_data()
    # Unconstrained base line - max_depth=None
    tree_unconstrained = DecisionTreeClassifier(random_state=42)
    tree_unconstrained.fit(X_train,y_train)

    train_auc_u = roc_auc_score(y_train, tree_unconstrained.predict_proba(X_train)[:, 1])
    val_auc_u = roc_auc_score(y_val, tree_unconstrained.predict_proba(X_val)[:,1])
    # Constrained baseline - max_depth = 3
    tree_constrained = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_constrained.fit(X_train, y_train)

    train_auc_c = roc_auc_score(
        y_train, tree_constrained.predict_proba(X_train)[:, 1]
    )
    val_auc_c = roc_auc_score(y_val, tree_constrained.predict_proba(X_val)[:, 1]
    )
    # constrained balanced baseline tree
    tree_balanced = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    tree_balanced.fit(X_train,y_train)
    train_auc_b = roc_auc_score(
        y_train, tree_balanced.predict_proba(X_train)[:, 1]
    )
    val_auc_b = roc_auc_score(
        y_val, tree_balanced.predict_proba(X_val)[:, 1]
    )

# Tabulted output
    summary_data = [
        [
            "Unconstrained (depth=None)",
            f"{train_auc_u:.4f}",
            f"{val_auc_u:.4f}",
            f"{train_auc_u - val_auc_u:+.4f}",
            "Severe Overfit",
        ],
        [
            "Shallow Tree (depth=3)",
            f"{train_auc_c:.4f}",
            f"{val_auc_c:.4f}",
            f"{train_auc_c - val_auc_c:+.4f}",
            "Controlled Baseline",
        ],
        [
            "Shallow Tree (depth=3, balanced)",
            f"{train_auc_b:.4f}",
            f"{val_auc_b:.4f}",
            f"{train_auc_b - val_auc_b:+.4f}",
            "Class Imbalance Adjusted",
        ],
    ]

    headers = ["Model Setup", "Train ROC-AUC", "Val ROC-AUC", "Gap", "Diagnosis"]

    print("\n--- BASELINE EVALUATION METRICS ---")
    print(tabulate(summary_data, headers=headers, tablefmt="fancy_grid"))

def tune_hyperparameters():
    # we will be using 5-fold cross validatiio optimization for ROC-AUC
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    base_tree = DecisionTreeClassifier(random_state=42)
    param_grid = {
    "max_depth": [3, 5, 8, 10],
    "min_samples_leaf": [5, 20],
    "criterion": ["gini"],
    "class_weight": ["balanced"],
    }
    grid_search= GridSearchCV(
        estimator=base_tree,
        scoring='roc_auc',
        param_grid=param_grid,
        cv= 5,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    # evaluating the best_model
    train_auc = roc_auc_score(
        y_train, best_model.predict_proba(X_train)[:, 1]
    )
    val_auc = roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])

    # tabulting it 
    tuning_summary = [
        [
            "Tuned Optimal Tree",
            f"{grid_search.best_score_:.4f}",
            f"{train_auc:.4f}",
            f"{val_auc:.4f}",
            f"{train_auc - val_auc:+.4f}",
        ]
    ]
    headers = [
        "Model",
        "CV Mean ROC-AUC",
        "Train ROC-AUC",
        "Val ROC-AUC",
        "Gap",
    ]
    print("\n--- TUNED MODEL PERFORMANCE SUMMARY ---")
    print(tabulate(tuning_summary, headers=headers, tablefmt="fancy_grid"))

    # Save trained model to disk
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "best_decision_tree.joblib"
    joblib.dump(best_model, str(model_path))    
    print(f"Saved optimal model to: {model_path}\n")

    return best_model

if __name__ == "__main__":
    baseline()
    tune_hyperparameters()