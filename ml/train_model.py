"""
train_model.py  —  Income Predictor ML Pipeline
================================================
Trains Ridge Regression + Decision Tree + Bagging Regressor on
Adult Census data with a statistically-grounded synthetic salary target.
Exports all model parameters as JSON so Node.js can run predictions
without any Python runtime dependency.
"""

import json, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ── 1. Load & clean data ──────────────────────────────────────────
df = pd.read_csv("../adult.csv")
df.columns = df.columns.str.strip()
for col in df.select_dtypes("object"):
    df[col] = df[col].str.strip().replace("?", np.nan)
df.dropna(inplace=True)

print(f"Dataset: {len(df):,} rows × {len(df.columns)} cols")
print("Columns:", list(df.columns))
print("Income distribution:\n", df["income"].value_counts())

# ── 2. Synthetic salary generation (statistically grounded) ───────
# We use per-occupation × education wage anchors + lognormal noise.
# This gives realistic, correlated salary values rather than random ranges.

OCCUPATION_ANCHOR = {
    "Exec-managerial":    105_000,
    "Prof-specialty":      98_000,
    "Tech-support":        62_000,
    "Sales":               55_000,
    "Craft-repair":        52_000,
    "Transport-moving":    46_000,
    "Machine-op-inspct":   40_000,
    "Protective-serv":     54_000,
    "Adm-clerical":        38_000,
    "Farming-fishing":     30_000,
    "Handlers-cleaners":   28_000,
    "Other-service":       26_000,
    "Priv-house-serv":     22_000,
    "Armed-Forces":        50_000,
}

def generate_salary(row):
    """
    Salary = base_anchor × edu_mult × hours_mult × noise
    Clamped to [$18K, $185K].  The income column ensures:
      <=50K rows → clamp upper end so salary ≤ 50K
      >50K  rows → floor at 51K
    """
    rng = np.random.default_rng(seed=int(row.name) % 100_000)

    occ_anchor = OCCUPATION_ANCHOR.get(str(row["occupation"]), 40_000)

    # Education multiplier: 8 yrs → 0.65, 13 yrs → 1.0, 16 yrs → 1.35
    edu_mult = 0.55 + (row["education.num"] / 13.0) * 0.70

    # Hours multiplier: 40 h → 1.0 baseline
    hours_mult = max(0.6, min(1.4, row["hours.per.week"] / 40.0))

    # Capital gains shift
    cap_bonus = min(0.25, row["capital.gain"] / 40_000.0)

    base = occ_anchor * edu_mult * hours_mult * (1 + cap_bonus)

    # Lognormal noise: σ = 0.18 ≈ ±18% typical wage spread
    noise = rng.lognormal(mean=0.0, sigma=0.18)
    salary = base * noise

    # Enforce income-class consistency
    if row["income"] == "<=50K":
        salary = min(salary, 50_000)
        salary = max(salary, 15_000)
    else:  # >50K
        salary = max(salary, 51_000)
        salary = min(salary, 185_000)

    return round(salary)

df["salary"] = df.apply(generate_salary, axis=1)

print(f"\nSalary stats:")
print(df["salary"].describe().apply(lambda x: f"${x:,.0f}"))

# ── 3. Feature engineering ────────────────────────────────────────
CATEGORICAL = ["workclass","education","marital.status","occupation",
               "relationship","race","sex","native.country"]
NUMERIC     = ["age","education.num","capital.gain","capital.loss","hours.per.week"]

# Label encode categoricals (store mappings for JS)
encoders = {}
df_enc = df.copy()
for col in CATEGORICAL:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = list(le.classes_)          # index → class name

FEATURES = NUMERIC + CATEGORICAL

X = df_enc[FEATURES].values.astype(float)
y = df_enc["salary"].values.astype(float)

# Scale numeric features
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[:, :len(NUMERIC)] = scaler.fit_transform(X[:, :len(NUMERIC)])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── 4. Train models ───────────────────────────────────────────────
print("\n── Training models ──")

# Ridge Regression
ridge = Ridge(alpha=10.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_mae  = mean_absolute_error(y_test, ridge_pred)
ridge_r2   = r2_score(y_test, ridge_pred)
print(f"Ridge   MAE=${ridge_mae:,.0f}  R²={ridge_r2:.3f}")

# Decision Tree
tree = DecisionTreeRegressor(max_depth=12, min_samples_leaf=20, random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_mae  = mean_absolute_error(y_test, tree_pred)
tree_r2   = r2_score(y_test, tree_pred)
print(f"DecTree MAE=${tree_mae:,.0f}  R²={tree_r2:.3f}")

# Bagging
bag = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=10, min_samples_leaf=15),
    n_estimators=50, max_samples=0.8, random_state=42, n_jobs=-1
)
bag.fit(X_train, y_train)
bag_pred = bag.predict(X_test)
bag_mae  = mean_absolute_error(y_test, bag_pred)
bag_r2   = r2_score(y_test, bag_pred)
print(f"Bagging MAE=${bag_mae:,.0f}  R²={bag_r2:.3f}")

# LDA (for classification component — we use it to classify <=50K vs >50K
#       to help calibrate the regression output)
y_class = (df_enc["income"].map({"<=50K": 0, ">50K": 1}).values
           if "income" in df_enc.columns
           else (y > 50_000).astype(int))
# Re-encode income if it hasn't been encoded yet
y_class = df["income"].map({"<=50K": 0, ">50K": 1}).values
_, _, y_cls_train, y_cls_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_cls_train)
lda_acc = (lda.predict(X_test) == y_cls_test).mean()
print(f"LDA     Acc={lda_acc:.3f}")

# ── 5. Compute residual statistics for confidence intervals ────────
# Bagging is our best model; use residuals to calibrate CIs
residuals = y_test - bag_pred
std_residual = np.std(residuals)
print(f"\nBagging residual std: ${std_residual:,.0f}")

# Per-income-class residual statistics (for tighter CIs)
low_mask  = y_test <= 50_000
high_mask = ~low_mask
std_low   = np.std(residuals[low_mask])   if low_mask.sum()  > 10 else std_residual
std_high  = np.std(residuals[high_mask])  if high_mask.sum() > 10 else std_residual
print(f"  ≤50K std: ${std_low:,.0f}   >50K std: ${std_high:,.0f}")

# ── 6. Export everything for Node.js ─────────────────────────────
def serialize_tree(tree_model, feature_names, depth=0):
    """Serialize a single sklearn DecisionTree into a JSON-friendly dict."""
    t = tree_model.tree_
    def recurse(node):
        if t.children_left[node] == -1:   # leaf
            return {"leaf": True, "value": float(t.value[node][0][0])}
        return {
            "leaf":    False,
            "feature": int(t.feature[node]),
            "threshold": float(t.threshold[node]),
            "left":    recurse(t.children_left[node]),
            "right":   recurse(t.children_right[node]),
        }
    return recurse(0)

# We export up to 3 of the 50 bagging estimators to keep JSON small
# but still useful for variance estimation.
N_TREES_EXPORT = 5
tree_list = [serialize_tree(est, FEATURES)
             for est in bag.estimators_[:N_TREES_EXPORT]]

model_data = {
    "meta": {
        "features":       FEATURES,
        "numeric_cols":   NUMERIC,
        "categorical_cols": CATEGORICAL,
        "n_train":        int(len(X_train)),
        "n_test":         int(len(X_test)),
    },
    "encoders": encoders,          # {"workclass": ["Federal-gov","Local-gov",...], ...}
    "scaler": {
        "mean": scaler.mean_.tolist(),
        "std":  scaler.scale_.tolist(),
    },
    "ridge": {
        "coef":      ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
        "mae":       round(ridge_mae),
        "r2":        round(ridge_r2, 4),
    },
    "tree": {
        "root": serialize_tree(tree, FEATURES),
        "mae":  round(tree_mae),
        "r2":   round(tree_r2, 4),
    },
    "bagging": {
        "trees":    tree_list,
        "n_trees_total": 50,
        "n_trees_exported": N_TREES_EXPORT,
        "mae":      round(bag_mae),
        "r2":       round(bag_r2, 4),
        "std_residual":  round(std_residual),
        "std_low":       round(std_low),
        "std_high":      round(std_high),
    },
    "lda": {
        "coef":      lda.coef_.tolist(),
        "intercept": lda.intercept_.tolist(),
        "accuracy":  round(lda_acc, 4),
    },
    "occupation_anchors": OCCUPATION_ANCHOR,
    "salary_stats": {
        "mean":   round(df["salary"].mean()),
        "median": round(df["salary"].median()),
        "std":    round(df["salary"].std()),
        "p25":    round(df["salary"].quantile(0.25)),
        "p75":    round(df["salary"].quantile(0.75)),
    }
}

out_path = "model_data.json"
with open(out_path, "w") as f:
    json.dump(model_data, f, separators=(",",":"))

size_kb = len(json.dumps(model_data)) / 1024
print(f"\n✅  Exported model_data.json  ({size_kb:.1f} KB)")
print("   Ridge  coefs:", len(model_data["ridge"]["coef"]))
print("   Bagging trees exported:", N_TREES_EXPORT)
