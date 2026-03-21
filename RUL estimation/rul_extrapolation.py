import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

    # -----------------------------
    # DATA PREP
    # -----------------------------
    cycle = df['Cycle_Number'].values
    capacity = df['Discharge_Capacity'].values

    capacity_norm = capacity / capacity[0]
    threshold = 0.8

    X = cycle.reshape(-1,1)
    y = capacity_norm

    future_cycles = np.arange(1, 300).reshape(-1,1)

    degrees = [1, 2, 3]

    # =============================
    # PLOT
    # =============================
    plt.figure(figsize=(8,5))

    plt.plot(cycle, capacity_norm, 'o', label='Actual')

    # =============================
    # MODELS
    # =============================
    for deg in degrees:

        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        future_poly = poly.transform(future_cycles)
        pred = model.predict(future_poly)

        # Find RUL (print only)
        try:
            rul_index = np.where(pred <= threshold)[0][0]
            rul = future_cycles[rul_index][0]
        except:
            rul = None

        print(f"{name} | Degree {deg} RUL:", rul)

        # Plot only curve
        plt.plot(future_cycles, pred, label=f'Poly deg {deg}')

    # Threshold line
    plt.axhline(y=threshold, linestyle='--', label='80% Threshold')

    # Labels
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.title(f"Capacity Fade Extrapolation - {name}")
    plt.legend()
    plt.grid()

    plt.show()

print("\nRUL extrapolation completed!")
