import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from scipy.signal import savgol_filter

# -----------------------------
# PATH
# -----------------------------
data_folder = "data/raw/"

files = glob(os.path.join(data_folder, "*.xlsx"))

# -----------------------------
# SELECTED CYCLES
# -----------------------------
cycles_to_plot = [1, 50, 99]

# -----------------------------
# FUNCTION: dV/dQ
# -----------------------------
def compute_dVdQ(df_cycle):

    df_cycle = df_cycle[df_cycle['Current(A)'] < 0]
    df_cycle = df_cycle.sort_values('Capacity(Ah)')

    V = df_cycle['Voltage(V)'].values
    Q = df_cycle['Capacity(Ah)'].values

    if len(V) < 10:
        return None, None

    dV = np.diff(V)
    dQ = np.diff(Q)

    dQ[dQ == 0] = 1e-6

    dVdQ = dV / dQ

    if len(dVdQ) > 11:
        dVdQ = savgol_filter(dVdQ, 11, 3)

    Q_mid = (Q[:-1] + Q[1:]) / 2

    return Q_mid, dVdQ


# =============================
# 1️⃣ SEPARATE PLOTS
# =============================
for file in files:

    print(f"\nProcessing: {file}")

    df = pd.read_excel(file)
    name = os.path.basename(file)

    plt.figure(figsize=(8,5))

    for c in cycles_to_plot:

        cycle_data = df[df['Cycle_Index'] == c]

        if len(cycle_data) == 0:
            print(f"Cycle {c} not found in {name}")
            continue

        Q_mid, dVdQ = compute_dVdQ(cycle_data)

        if Q_mid is None:
            continue

        plt.plot(Q_mid, dVdQ, label=f'Cycle {c}')

    plt.xlabel("Capacity (Ah)")
    plt.ylabel("dV/dQ")
    plt.title(f"dV/dQ (RAW) - {name}")
    plt.legend()
    plt.grid(True)

    plt.show()


# =============================
# 2️⃣ COMBINED PLOT
# =============================
plt.figure(figsize=(8,5))

for file in files:

    df = pd.read_excel(file)
    name = os.path.basename(file)

    cycle_data = df[df['Cycle_Index'] == 2]

    if len(cycle_data) == 0:
        continue

    Q_mid, dVdQ = compute_dVdQ(cycle_data)

    if Q_mid is None:
        continue

    plt.plot(Q_mid, dVdQ, label=name)

plt.xlabel("Capacity (Ah)")
plt.ylabel("dV/dQ")
plt.title("dV/dQ Comparison (Cycle 2)")
plt.legend()
plt.grid(True)

plt.show()
