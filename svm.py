import os
import numpy as np
import pandas as pd
import joblib

# Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

# SVM & Ensemble for Speed
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

import category_encoders as ce

# NOTE: SVM in scikit-learn is CPU-only. 
# We import torch just to maintain file structure compatibility if needed, 
# but it is not used for computation here.
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device detected (unused by sklearn SVM):", DEVICE)

# ==========================================
# CONFIG
# ==========================================
RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "RiskFlag"
ID_COL = "ProfileID"

os.makedirs("submissions", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("oof_preds", exist_ok=True)

# ==========================================
# FIND FILES
# ==========================================
def find_file(fn):
    base = "/kaggle/input/risked"
    for root, dirs, files in os.walk(base):
        if fn in files:
            return os.path.join(root, fn)
    raise FileNotFoundError(f"{fn} not found under {base}")

TRAIN_PATH = find_file("train_updated.csv")
TEST_PATH  = find_file("test_updated.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape, " Test shape:", test.shape)

# ==========================================
# TARGET & FEATURES
# ==========================================
y = train[TARGET].astype(int).fillna(0)

feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# ==========================================
# FEATURE ENGINEERING
# ==========================================
# 1) Numeric Aggregations
if numeric_cols:
    X["num_mean"] = X[numeric_cols].mean(axis=1)
    X["num_std"]  = X[numeric_cols].std(axis=1).fillna(0)
    X["num_min"]  = X[numeric_cols].min(axis=1)
    X["num_max"]  = X[numeric_cols].max(axis=1)
    
    X_test["num_mean"] = X_test[numeric_cols].mean(axis=1)
    X_test["num_std"]  = X_test[numeric_cols].std(axis=1).fillna(0)
    X_test["num_min"]  = X_test[numeric_cols].min(axis=1)
    X_test["num_max"]  = X_test[numeric_cols].max(axis=1)

    agg_feats = ["num_mean", "num_std", "num_min", "num_max"]
    numeric_cols = numeric_cols + agg_feats

# 2) Frequency Encoding
for c in cat_cols:
    freq = X[c].value_counts(normalize=True)
    X[c + "_freq"] = X[c].map(freq)
    X_test[c + "_freq"] = X_test[c].map(freq).fillna(0)
    numeric_cols.append(c + "_freq")

# 3) Cardinality Groups
low_card_cat = [c for c in cat_cols if X[c].nunique() <= 10]
high_card_cat = [c for c in cat_cols if X[c].nunique() > 10]

# Drop constant columns
const_cols = [c for c in numeric_cols if X[c].nunique() <= 1]
if const_cols:
    print("Dropping constant numeric columns:", const_cols)
    X = X.drop(columns=const_cols)
    X_test = X_test.drop(columns=const_cols)
    numeric_cols = [c for c in numeric_cols if c not in const_cols]

# ==========================================
# PREPROCESSOR (Quantile is best for SVM)
# ==========================================
def make_preprocessor(method='quantile'):
    num_steps = [('imp', SimpleImputer(strategy='median'))]
    
    if method == 'quantile':
        # SVM works significantly better if data is Normal (Gaussian)
        num_steps.append(('quant', QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)))
    elif method == 'scale':
        num_steps.append(('sc', StandardScaler()))

    num_pipe = Pipeline(num_steps)

    low_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    high_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('te', ce.TargetEncoder())
    ])

    transformers = []
    if numeric_cols:
        transformers.append(('num', num_pipe, numeric_cols))
    if low_card_cat:
        transformers.append(('low', low_pipe, low_card_cat))
    if high_card_cat:
        transformers.append(('high', high_pipe, high_card_cat))

    return ColumnTransformer(transformers)

# ==========================================
# MODEL: BAGGED SVM (The Speed Solution)
# ==========================================
# Using BaggingClassifier with SVC allows us to train on smaller subsets 
# of data in parallel. 
# Complexity drops from O(N^3) to roughly O(N^3 / k^2) where k is splits.

fast_svm_estimator = BaggingClassifier(
    estimator=SVC(
        kernel='rbf',
        C=1.5,
        probability=True,        # Required for predict_proba
        cache_size=2000,         # Use 2GB RAM for cache to speed up
        decision_function_shape='ovr'
    ),
    n_estimators=10,             # Train 10 independent SVMs
    max_samples=0.15,            # Each SVM sees only 15% of the data (Speed hack!)
    bootstrap=True,              # Standard bagging
    n_jobs=-1,                   # RUN IN PARALLEL on all CPU cores
    random_state=RANDOM_STATE
)

models = {
    "svm_bagged_fast": {
        "estimator": fast_svm_estimator,
        "scale_method": "quantile"
    }
}

print("Selected Model:", list(models.keys()))

# ==========================================
# HELPER: threshold search
# ==========================================
def find_best_threshold(y_true, probs):
    best_thr = 0.5
    best_acc = 0.0
    for thr in np.linspace(0.2, 0.8, 601):
        preds = (probs >= thr).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc

# ==========================================
# CV LOOP
# ==========================================
def run_cv(name, cfg):
    print(f"\n==== TRAINING: {name} ====")
    print("Note: Even with optimization, SVM is slower than LogReg.")
    print("Bagging is running 10 SVMs in parallel on 15% data each...")
    
    pre = make_preprocessor(cfg["scale_method"])
    base_model = cfg["estimator"]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros(len(X), dtype=float)
    test_preds_fold = np.zeros((N_SPLITS, len(X_test)), dtype=float)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}...")
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        model = clone(base_model)
        pipe = Pipeline([("pre", pre), ("clf", model)])
        
        pipe.fit(X_tr, y_tr)

        # Predict
        oof[va] = pipe.predict_proba(X_va)[:, 1]
        test_preds_fold[fold - 1] = pipe.predict_proba(X_test)[:, 1]

    best_thr, best_acc = find_best_threshold(y, oof)
    print(f"  >> {name} Best Threshold: {best_thr:.4f} | OOF Accuracy: {best_acc:.6f}")

    pd.DataFrame({
        ID_COL: train[ID_COL],
        "oof_proba": oof,
        "oof_pred": (oof >= best_thr).astype(int)
    }).to_csv(f"oof_preds/{name}.csv", index=False)

    avg_test_prob = test_preds_fold.mean(axis=0)
    final_pred = (avg_test_prob >= best_thr).astype(int)

    pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET: final_pred
    }).to_csv(f"submissions/{name}.csv", index=False)

    full_model = clone(base_model)
    full_pipe = Pipeline([("pre", pre), ("clf", full_model)])
    full_pipe.fit(X, y)
    joblib.dump(full_pipe, f"models/{name}.joblib")

    return best_acc

# ==========================================
# EXECUTION
# ==========================================
results = {}

for name, cfg in models.items():
    try:
        acc = run_cv(name, cfg)
        results[name] = acc
    except Exception as e:
        print(f"Failed to run {name}: {e}")

print("\nFINAL SVM RESULTS:")
for k, v in results.items():
    print(f"{k}: {v:.6f}")