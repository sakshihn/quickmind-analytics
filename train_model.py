"""
AI-Powered Quick Commerce Delivery Intelligence System
Model Training Script
Author: Final Year Data Science Project
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  QUICK COMMERCE AI — MODEL TRAINING PIPELINE")
print("=" * 60)

# ── 1. Load dataset ──────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "cleaned_quick_commerce.csv")
print(f"\n[1/7] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"      Rows: {len(df):,}  |  Columns: {df.shape[1]}")

# ── 2. Feature Engineering ───────────────────────────────────
print("\n[2/7] Engineering features...")
df['SLA_Breach'] = (df['Delivery_Time'] > 30).astype(int)
df['Revenue_Tier'] = pd.cut(df['Order_Value'],
                             bins=[0, 200, 600, 1200, 99999],
                             labels=['Low', 'Medium', 'High', 'Premium'])
df['Distance_Bin'] = pd.cut(df['Distance_km'],
                              bins=[0, 5, 10, 20, 100],
                              labels=['Very_Short', 'Short', 'Medium', 'Long'])
print("      SLA_Breach, Revenue_Tier, Distance_Bin added.")

# ── 3. Define features & target ──────────────────────────────
print("\n[3/7] Defining features and target variable...")
FEATURES = ['Distance_km', 'Order_Value', 'Customer_Age', 'Items_Count',
            'Product_Category', 'Payment_Method', 'Company', 'City',
            'Discount_Applied', 'Delivery_Partner_Rating']
TARGET = 'Delivery_Time'

X = df[FEATURES].copy()
y = df[TARGET].copy()
print(f"      Features: {len(FEATURES)}  |  Target: {TARGET}")

# ── 4. Encode categoricals ───────────────────────────────────
print("\n[4/7] Encoding categorical features...")
encoders = {}
cat_cols = ['Product_Category', 'Payment_Method', 'Company', 'City']
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le
    print(f"      Encoded: {col}  ({len(le.classes_)} classes)")

# Save encoders
os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
joblib.dump(encoders, os.path.join(os.path.dirname(__file__), "models", "label_encoders.pkl"))
joblib.dump(FEATURES, os.path.join(os.path.dirname(__file__), "models", "feature_names.pkl"))

# ── 5. Train/Test split ──────────────────────────────────────
print("\n[5/7] Splitting data (80/20 train-test)...")
# Sample for faster training; use full set for production
SAMPLE_SIZE = 150_000
if len(df) > SAMPLE_SIZE:
    print(f"      Using {SAMPLE_SIZE:,} rows sample for training speed.")
    X_s, y_s = X.sample(SAMPLE_SIZE, random_state=42), y.loc[X.sample(SAMPLE_SIZE, random_state=42).index]
    X_s.reset_index(drop=True, inplace=True)
    y_s.reset_index(drop=True, inplace=True)
else:
    X_s, y_s = X, y

X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.2, random_state=42)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae:.4f} mins")
    print(f"    RMSE : {rmse:.4f} mins")
    print(f"    R²   : {r2:.4f}")
    return model, mae, rmse, r2

# ── 6. Train models ──────────────────────────────────────────
print("\n[6/7] Training models...")

lr, lr_mae, lr_rmse, lr_r2 = evaluate(
    "Linear Regression",
    LinearRegression(),
    X_train, X_test, y_train, y_test
)

rf, rf_mae, rf_rmse, rf_r2 = evaluate(
    "Random Forest",
    RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    X_train, X_test, y_train, y_test
)

xgb, xgb_mae, xgb_rmse, xgb_r2 = evaluate(
    "XGBoost",
    XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1,
                 random_state=42, n_jobs=-1, verbosity=0),
    X_train, X_test, y_train, y_test
)

# ── 7. Pick best and save ────────────────────────────────────
print("\n[7/7] Selecting best model and saving artifacts...")
results = {
    'Linear Regression': (lr, lr_mae, lr_r2),
    'Random Forest':     (rf, rf_mae, rf_r2),
    'XGBoost':           (xgb, xgb_mae, xgb_r2),
}
best_name = min(results, key=lambda k: results[k][1])
best_model = results[best_name][0]
print(f"\n  ✅  Best Model: {best_name}  (MAE={results[best_name][1]:.4f}, R²={results[best_name][2]:.4f})")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
joblib.dump(best_model,   os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(lr,           os.path.join(MODEL_DIR, "linear_regression.pkl"))
joblib.dump(rf,           os.path.join(MODEL_DIR, "random_forest.pkl"))
joblib.dump(xgb,          os.path.join(MODEL_DIR, "xgboost.pkl"))

# Save comparison metrics
metrics = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE':   [results[k][1] for k in results],
    'R2':    [results[k][2] for k in results],
}).round(4)
metrics.to_csv(os.path.join(MODEL_DIR, "model_metrics.csv"), index=False)

print("\n  Saved:")
print(f"    → {MODEL_DIR}/best_model.pkl")
print(f"    → {MODEL_DIR}/label_encoders.pkl")
print(f"    → {MODEL_DIR}/model_metrics.csv")
print("\n" + "=" * 60)
print("  TRAINING COMPLETE — Run app.py to launch the dashboard")
print("=" * 60)