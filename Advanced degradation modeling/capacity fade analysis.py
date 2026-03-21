import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from glob import glob

# -----------------------------
# FOLDER PATH
# -----------------------------
folder_path = "data/Processed/"

# Get all Excel files
files = glob(os.path.join(folder_path, "*.xlsx"))

# -----------------------------
# MODEL FUNCTIONS
# -----------------------------
def exponential_model(n, k):
    return np.exp(-k * n)

def sqrt_model(n, k):
    return 1 - k * np.sqrt(n)

# -----------------------------
# ERROR FUNCTION
# -----------------------------
def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))

# -----------------------------
# PROCESS ALL FILES
# -----------------------------
for file in files:

    print(f"\nProcessing: {file}")

    df = pd.read_excel(file)

    cycle = df['Cycle_Number']
    capacity = df['Discharge_Capacity']

    capacity_norm = capacity / capacity.iloc[0]

    # Fit models
    params_exp, _ = curve_fit(exponential_model, cycle, capacity_norm)
    params_sqrt, _ = curve_fit(sqrt_model, cycle, capacity_norm)

    exp_fit = exponential_model(cycle, *params_exp)
    sqrt_fit = sqrt_model(cycle, *params_sqrt)

    # Errors
    print(f"Exponential RMSE: {rmse(capacity_norm, exp_fit):.5f}")
    print(f"Square Root RMSE: {rmse(capacity_norm, sqrt_fit):.5f}")

    # Plot
    plt.figure()

    plt.plot(cycle, capacity_norm, 'o', label='Actual')
    plt.plot(cycle, exp_fit, '-', label='Exponential')
    plt.plot(cycle, sqrt_fit, '--', label='Square Root')

    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.title(os.path.basename(file))
    plt.legend()

    plt.show()

print("\nAll files processed!")
