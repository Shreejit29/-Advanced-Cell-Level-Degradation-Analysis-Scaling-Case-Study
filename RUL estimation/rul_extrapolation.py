import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# =============================
# SETTINGS
# =============================
threshold = 0.8
max_cycle_limit = 100

# -----------------------------
# COLOR MAP
# -----------------------------
colors = {
    1: 'blue',
    2: 'green',
    3: 'orange'
}

# =============================
# PATH
# =============================
data_folder = "data/processed/"


files = glob(os.path.join(data_folder, "*.xlsx"))

print("Files found:", files)

# =============================
# LOOP THROUGH DATASETS
# =============================
for file in files:

    print("\n" + "="*50)
    print(f"Processing: {file}")

    df = pd.read_excel(file)
    name = os.path.basename(file).replace(".xlsx", "")

    # -----------------------------
    # DATA PREP
    # -----------------------------
    cycle = df['Cycle_Number'].values
    capacity = df['Discharge_Capacity'].values
    capacity_norm = capacity / capacity[0]

    # =============================
    # ACTUAL FAILURE
    # =============================
    try:
        actual_index = np.where(capacity_norm <= threshold)[0][0]
        actual_failure = cycle[actual_index]
    except:
        actual_failure = None

    print(f"Actual Failure Cycle: {actual_failure}")

    # =============================
    # EARLY TRAINING
    # =============================
    if actual_failure is not None:
        cutoff_cycle = int(0.6 * actual_failure)
    else:
        cutoff_cycle = int(0.6 * len(cycle))

    mask = cycle <= cutoff_cycle
    cycle_early = cycle[mask]
    cap_early = capacity_norm[mask]

    current_cycle = cycle_early[-1]

    print(f"Training up to cycle: {current_cycle}")

    # =============================
    # FUTURE RANGE (UP TO 100)
    # =============================
    future_cycles = np.arange(current_cycle + 1, max_cycle_limit + 1)

    # =============================
    # MODEL INPUT
    # =============================
    X = cycle_early.reshape(-1,1)
    y = cap_early

    degrees = [1, 2, 3]

    # =============================
    # PLOT SETUP
    # =============================
    plt.figure(figsize=(8,5))

    # Actual data
    plt.plot(cycle, capacity_norm, 'o', color='black', label='Actual')

    # =============================
    # MODELS
    # =============================
    for deg in degrees:

        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        # Combine early + future
        full_cycles = np.concatenate([cycle_early, future_cycles])
        full_cycles_reshaped = full_cycles.reshape(-1,1)

        full_poly = poly.transform(full_cycles_reshaped)
        pred = model.predict(full_poly)

        # -----------------------------
        # FIND FAILURE
        # -----------------------------
        predicted_failure = None
        rul = None
        error = None

        for i, val in enumerate(pred):
            if val <= threshold:
                predicted_failure = full_cycles[i]
                rul = predicted_failure - current_cycle

                if actual_failure is not None:
                    error = abs(predicted_failure - actual_failure)
                break

        # -----------------------------
        # PRINT RESULTS
        # -----------------------------
        print(f"\n{name} | Degree {deg}")
        print(f"Predicted Failure Cycle: {predicted_failure}")
        print(f"RUL (Remaining Cycles): {rul}")
        print(f"Prediction Error: {error}")

        # -----------------------------
        # PLOT MODEL
        # -----------------------------
        plt.plot(full_cycles, pred,
                 color=colors[deg],
                 linewidth=2,
                 label=f'Degree {deg}')

        # Predicted failure marker
        if predicted_failure is not None:
            plt.axvline(x=predicted_failure,
                        color=colors[deg],
                        linestyle='--',
                        alpha=0.7)

    # -----------------------------
    # THRESHOLD
    # -----------------------------
    plt.axhline(y=threshold,
                color='red',
                linestyle='--',
                linewidth=2,
                label='80% Threshold')

    # Actual failure
    if actual_failure is not None:
        plt.axvline(x=actual_failure,
                    color='red',
                    linestyle='-',
                    linewidth=2,
                    label='Actual Failure')

    # Training cutoff
    plt.axvline(x=current_cycle,
                color='purple',
                linestyle=':',
                linewidth=2,
                label='Training Cutoff')

    # -----------------------------
    # FINAL SETTINGS
    # -----------------------------
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.title(f"Capacity Extrapolation (Early Prediction) - {name}")
    plt.legend(loc='best', fontsize=9)
    plt.grid()

  
    plt.show()

print("\n✅ Final Capacity Extrapolation Completed!")
