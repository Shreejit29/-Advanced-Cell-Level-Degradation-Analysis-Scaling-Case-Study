import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# -----------------------------
# PARAMETERS
# -----------------------------
k = 5
threshold = 0.8
future_max = 300

# -----------------------------
# PATH
# -----------------------------
data_folder = "data/processed/"
files = glob(os.path.join(data_folder, "*.xlsx"))

print("Files found:", files)

# =============================
# LOOP THROUGH DATASETS
# =============================
for file in files:

    print(f"\nProcessing: {file}")

    df = pd.read_excel(file)
    name = os.path.basename(file)

    # =============================
    # FEATURE ENGINEERING
    # =============================
    df['CEF_exp'] = df['Coulombic_Efficiency'] * np.exp(-k * (1 - df['Energy_Efficiency']))

    X = df[['Coulombic_Efficiency', 'Energy_Efficiency', 'CEF_exp']]
    y = df['Discharge_Capacity'] / df['Discharge_Capacity'].iloc[0]

    cycle = df['Cycle_Number'].values

    # =============================
    # FUTURE DATA (UP TO 300)
    # =============================
    last_row = df.iloc[-1]

    future_cycles = np.arange(cycle[-1] + 1, future_max + 1)

    future_df = pd.DataFrame({
        'Coulombic_Efficiency': [last_row['Coulombic_Efficiency']] * len(future_cycles),
        'Energy_Efficiency': [last_row['Energy_Efficiency']] * len(future_cycles)
    })

    future_df['CEF_exp'] = future_df['Coulombic_Efficiency'] * np.exp(-k * (1 - future_df['Energy_Efficiency']))

    X_future = future_df

    # =============================
    # MODELS
    # =============================
    models = {
        "Linear": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    # =============================
    # TRAIN + DISPLAY PLOTS
    # =============================
    for model_name, model in models.items():

        model.fit(X, y)

        # Predict past + future
        pred_past = model.predict(X)
        pred_future = model.predict(X_future)

        pred_full = np.concatenate([pred_past, pred_future])
        full_cycles = np.concatenate([cycle, future_cycles])

        # -----------------------------
        # FIND RUL (PRINT ONLY)
        # -----------------------------
        rul = None
        for i, val in enumerate(pred_full):
            if val <= threshold:
                rul = full_cycles[i]
                break

        print(f"{name} | {model_name} RUL:", rul)

        # =============================
        # 📊 PLOT (NO RUL LINE)
        # =============================
        plt.figure(figsize=(8,5))

        plt.plot(cycle, y, label='Actual')
        plt.plot(full_cycles, pred_full, label=model_name)

        plt.axhline(y=threshold, linestyle='--', label='80% Threshold')

        plt.xlabel("Cycle Number")
        plt.ylabel("Normalized Capacity")
        plt.title(f"{model_name} Prediction (300 cycles) - {name}")
        plt.legend()
        plt.grid()

        plt.show()

print("\nRUL ML (CEF-based) completed!")
