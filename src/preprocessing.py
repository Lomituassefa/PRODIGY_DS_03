import sys
from pathlib import Path

# Append parent directory (PRODIGY_DS_03) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from config import DATA_DIR

def load_data():
    df = pd.read_csv(DATA_DIR / "bank-additional-full.csv", sep=";")
    df['target'] = (df['y']=='yes').astype(int)
    df = df.drop(columns= ['y','duration'])
    df["was_pcontacted"] = (df["pdays"] != 999).astype(int)
    categorical_cols = [
        "job", "marital", "education", "default", 
        "housing", "loan", "contact", "month", 
        "day_of_week", "poutcome"
    ]
    numerical_cols = [
        "age","campaign","pdays","previous",
        "emp.var.rate","cons.price.idx","cons.conf.idx",
        "euribor3m","nr.employed","was_pcontacted",
    ]
    features = categorical_cols + numerical_cols
    X= df[features]
    y= df['target']
    # Stratified split
    X_train, X_test,y_train, y_test = train_test_split(
        X,y, test_size=0.2,random_state=42, stratify=y)
    X_train, X_val, y_train, y_val= train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)],
        remainder="passthrough"
    )
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    return X_train_proc, X_val_proc, X_test_proc, y_train, y_val, y_test

if __name__ == "__main__":
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_data()
    print("Preprocessing successful!")
    print(
        f"Train shape: {X_tr.shape} | Val shape: {X_va.shape} | Test shape:"
        f" {X_te.shape}",
    )