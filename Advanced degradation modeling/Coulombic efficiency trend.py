import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

# -----------------------------
# PATHS (CHANGE IF NEEDED)
# -----------------------------
data_folder = "data\Processed"

files = glob(os.path.join(data_folder, "*.xlsx"))

print("Files found:", files)

# =============================
# 1️⃣ SEPARATE PLOTS
# =============================
for file in files:

    df = pd.read_excel(file)
    name = os.path.basename(file).replace(".xlsx", "")

    plt.figure(figsize=(8,5))

    plt.plot(df['Cycle_Number'], df['Coulombic_Efficiency'], marker='o')

    plt.xlabel("Cycle Number")
    plt.ylabel("Coulombic Efficiency")
    plt.title(f"Coulombic Efficiency Trend - {name}")
    plt.grid()

    
    plt.show()


# =============================
# 2️⃣ COMBINED PLOT
# =============================
plt.figure(figsize=(8,5))

for file in files:

    df = pd.read_excel(file)
    name = os.path.basename(file).replace(".xlsx", "")

    plt.plot(df['Cycle_Number'], df['Coulombic_Efficiency'], marker='o', label=name)

plt.xlabel("Cycle Number")
plt.ylabel("Coulombic Efficiency")
plt.title("Coulombic Efficiency Comparison")
plt.legend()
plt.grid()
plt.show(

plt.show()
