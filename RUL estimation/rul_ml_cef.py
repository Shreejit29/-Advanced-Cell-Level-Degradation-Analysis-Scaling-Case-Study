import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from glob import glob
from sklearn.linear_model import LinearRegression

# =============================
# PARAMETERS
# =============================
threshold = 0.8
train_cycles = 5   # you can change (5, 10, 15)
max_cycle_limit = 100

# =============================
# PATH
# =============================
data_folder = "data/processed/"


files = glob(os.path.join(data_folder, "*.xlsx"))

print("Files found:", files)

# =============================
# LOOP
# =============================
for file in files:

    print("\n" + "="*70)
    print(f"Processing: {file}")

    df = pd.read_excel(file)
    name = os.path.basename(file).replace(".xlsx", "")

    # =============================
    # CEF
    # =============================
    df['CEF'] = 2 / (
        np.exp(10 * (1 - df['Coulombic_Efficiency'])) +
        np.exp(10 * (1 - df['Energy_Efficiency']))
    )

    # =============================
    # CAPACITY
    # =============================
    df['Capacity_norm'] = df['Discharge_Capacity'] / df['Discharge_Capacity'].iloc[0]

    cycle = df['Cycle_Number'].values
    capacity = df['Capacity_norm'].values

    # =============================
    # ACTUAL FAILURE
    # =============================
    try:
        actual_idx = np.where(capacity <= threshold)[0][0]
        actual_failure = int(cycle[actual_idx])
    except:
        print("⚠️ No failure found → skipping")
        continue

    # =============================
    # TRAIN DATA
    # =============================
    train_df = df[df['Cycle_Number'] <= train_cycles]

    if len(train_df) < 3:
        print("⚠️ Not enough training data")
        continue

    current_cycle = int(train_df['Cycle_Number'].iloc[-1])

    print(f"\nTraining up to cycle: {current_cycle}")
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
    future_cycles = np.arange(current_cycle + 1, max_cycle_limit + 1)

    cef_last = train_df['CEF'].iloc[-1]
    decay_rate = 0.03
    cef_future = cef_last * np.exp(-decay_rate * np.arange(len(future_cycles)))

    X_future = np.column_stack((future_cycles, cef_future))

    pred_past = model.predict(X_train)
    pred_future = model.predict(X_future)

    full_cycles = np.concatenate([train_df['Cycle_Number'], future_cycles])
    pred_full = np.concatenate([pred_past, pred_future])

    # =============================
    # FIND FAILURE
    # =============================
    predicted_failure = None

    for i, val in enumerate(pred_full):
        if val <= threshold:
            predicted_failure = int(full_cycles[i])
            break

    if predicted_failure is None:
        print("⚠️ No failure predicted")
        continue

    predicted_rul = predicted_failure - current_cycle
    error = abs(predicted_failure - actual_failure)

    # =============================
    # PRINT RESULTS
    # =============================
    print(f"\n{name} | Linear Regression")
    print(f"Predicted Failure Cycle: {predicted_failure}")
    print(f"RUL: {predicted_rul}")
    print(f"Error: {error}")

    # =============================
    # 🔥 PLOT 
    # =============================
    plt.figure(figsize=(8,5))

    plt.plot(cycle, capacity, 'o', color='black', label='Actual')
    plt.plot(full_cycles, pred_full, color='blue', label='Predicted')

    plt.axhline(y=threshold, color='red', linestyle='--', label='80% Threshold')
    plt.axvline(x=current_cycle, color='purple', linestyle=':', label='Training Cutoff')
    plt.axvline(x=predicted_failure, color='blue', linestyle='--', label='Predicted Failure')
    plt.axvline(x=actual_failure, color='red', linestyle='-', label='Actual Failure')

    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.title(f"Linear Model Capacity Prediction - {name}")
    plt.legend()
    plt.grid()

    plt.close()

    # =============================
    # 🔥 SHAP ANALYSIS (LINEAR)
    # =============================
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)

    print("\nFeature Importance (Mean |SHAP|):")

    feature_names = ['Cycle_Number', 'CEF']

    for i, feat in enumerate(feature_names):
        importance = np.mean(np.abs(shap_values[:, i]))
        print(f"{feat}: {importance:.4f}")

    # =============================
    # SHAP PLOT
    # =============================
    plt.figure()

    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=feature_names,
        show=False
    )

  
    plt.close()

print("\n✅ LINEAR MODEL + PLOTS + SHAP COMPLETE!")
