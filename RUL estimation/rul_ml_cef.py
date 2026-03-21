"""
Linear Regression Based Battery RUL Prediction (Clean Version)

- Input Features: Cycle_Number, CEF
- Target: Capacity_norm
- Method: Predict capacity → detect failure (80%)
- Explainability: SHAP (printed only)

Author: Prathmesh Udekar
"""

import pandas as pd
import numpy as np
import shap
import os
from glob import glob
from sklearn.linear_model import LinearRegression

# =============================
# PARAMETERS
# =============================
THRESHOLD = 0.8
TRAIN_CYCLES = 10
MAX_CYCLE_LIMIT = 100

# =============================
# PATH
# =============================
DATA_FOLDER = "data/processed/"
files = glob(os.path.join(DATA_FOLDER, "*.xlsx"))

if not files:
    raise FileNotFoundError("No dataset files found.")

print("Files found:", files)

# =============================
# MAIN LOOP
# =============================
for file in files:

    print("\n" + "="*60)
    print(f"Processing: {file}")

    df = pd.read_excel(file)
    name = os.path.basename(file).replace(".xlsx", "")

    # =============================
    # FEATURE: CEF
    # =============================
    df['CEF'] = 2 / (
        np.exp(10 * (1 - df['Coulombic_Efficiency'])) +
        np.exp(10 * (1 - df['Energy_Efficiency']))
    )

    # =============================
    # TARGET: NORMALIZED CAPACITY
    # =============================
    df['Capacity_norm'] = df['Discharge_Capacity'] / df['Discharge_Capacity'].iloc[0]

    cycle = df['Cycle_Number'].values
    capacity = df['Capacity_norm'].values

    # =============================
    # ACTUAL FAILURE
    # =============================
    try:
        actual_idx = np.where(capacity <= THRESHOLD)[0][0]
        actual_failure = int(cycle[actual_idx])
    except:
        print("No failure detected → skipping")
        continue

    # =============================
    # TRAIN DATA
    # =============================
    train_df = df[df['Cycle_Number'] <= TRAIN_CYCLES]

    if len(train_df) < 3:
        print("Not enough training data → skipping")
        continue

    current_cycle = int(train_df['Cycle_Number'].iloc[-1])

    print(f"\nTraining up to cycle: {current_cycle}")
    print(f"Training samples: {len(train_df)}")
    print(f"Actual Failure Cycle: {actual_failure}")

    # =============================
    # FEATURES
    # =============================
    X_train = train_df[['Cycle_Number', 'CEF']]
    y_train = train_df['Capacity_norm']

    # =============================
    # MODEL
    # =============================
    model = LinearRegression()
    model.fit(X_train, y_train)

    # =============================
    # FUTURE PREDICTION
    # =============================
    future_cycles = np.arange(current_cycle + 1, MAX_CYCLE_LIMIT + 1)

    cef_last = train_df['CEF'].iloc[-1]
    decay_rate = 0.03
    cef_future = cef_last * np.exp(-decay_rate * np.arange(len(future_cycles)))

    X_future = np.column_stack((future_cycles, cef_future))

    pred_past = model.predict(X_train)
    pred_future = model.predict(X_future)

    full_cycles = np.concatenate([train_df['Cycle_Number'], future_cycles])
    pred_full = np.concatenate([pred_past, pred_future])

    # =============================
    # FAILURE DETECTION
    # =============================
    predicted_failure = None

    for i, val in enumerate(pred_full):
        if val <= THRESHOLD:
            predicted_failure = int(full_cycles[i])
            break

    if predicted_failure is None:
        print("\n⚠️ No failure predicted")
        continue

    predicted_rul = predicted_failure - current_cycle
    error = abs(predicted_failure - actual_failure)

    # =============================
    # PRINT RESULTS
    # =============================
    print("\n----------------------------")
    print(f"{name} | Linear Regression")
    print(f"Predicted Failure Cycle: {predicted_failure}")
    print(f"RUL: {predicted_rul}")
    print(f"Error: {error}")

    # =============================
    # SHAP ANALYSIS (PRINT ONLY)
    # =============================
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)

    feature_names = ['Cycle_Number', 'CEF']
    importance = np.mean(np.abs(shap_values), axis=0)

    print("\nFeature Importance (Mean |SHAP|):")
    for f, imp in zip(feature_names, importance):
        print(f"{f}: {imp:.4f}")

print("\n✅ CLEAN EXECUTION COMPLETE (NO FILES SAVED)")
