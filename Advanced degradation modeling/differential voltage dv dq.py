import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from scipy.signal import savgol_filter


# PATHS
data_folder ="data/Processed/"

files = glob(os.path.join(data_folder, "*.xlsx"))

print("Files found:", files)

# SETTINGS
cycles_to_plot = [3, 50, 98]


# FUNCTION
def compute_dVdQ(df_cycle):

    # keep only discharge
    df_cycle = df_cycle[df_cycle['Current(A)'] < -0.05]

    # keep increasing capacity
    df_cycle = df_cycle[df_cycle['Capacity(Ah)'].diff().fillna(0) > 0]

    df_cycle = df_cycle.sort_values('Capacity(Ah)')

    V = df_cycle['Voltage(V)'].values
    Q = df_cycle['Capacity(Ah)'].values

    if len(V) < 20:
        return None, None

    dV = np.diff(V)
    dQ = np.diff(Q)

    valid = np.abs(dQ) > 1e-4
    dV = dV[valid]
    dQ = dQ[valid]

    if len(dV) < 10:
        return None, None

    dVdQ = dV / dQ
    dVdQ = np.clip(dVdQ, -10, 10)

    # smooth
    if len(dVdQ) > 11:
        dVdQ = savgol_filter(dVdQ, 11, 3)

    Q_mid = (Q[:-1] + Q[1:]) / 2
    Q_mid = Q_mid[valid]

    return Q_mid, dVdQ


# SEPARATE PLOTS

for file in files:

    print(f"\nProcessing: {file}")

    df = pd.read_excel(file)
    name = os.path.basename(file).replace(".xlsx", "")

    plt.figure(figsize=(8,5))

    for c in cycles_to_plot:

        cycle_data = df[df['Cycle_Index'] == c]

        if len(cycle_data) == 0:
            continue

        Q_mid, dVdQ = compute_dVdQ(cycle_data)

        if Q_mid is None:
            continue

        plt.plot(Q_mid, dVdQ, label=f'Cycle {c}')

    plt.xlabel("Capacity (Ah)")
    plt.ylabel("dV/dQ")
    plt.title(f"{name} - dV/dQ")
    plt.legend()
    plt.grid()

    plt.show()


# COMBINED PLOT

plt.figure(figsize=(8,5))

for file in files:

    df = pd.read_excel(file)
    name = os.path.basename(file).replace(".xlsx", "")

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
plt.grid()

plt.show()

