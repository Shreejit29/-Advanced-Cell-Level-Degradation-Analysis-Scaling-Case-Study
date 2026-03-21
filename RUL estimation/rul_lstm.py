import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# PARAMETERS
# -----------------------------
window = 5
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

    # -----------------------------
    # DATA PREP
    # -----------------------------
    cycle = df['Cycle_Number'].values
    capacity = df['Discharge_Capacity'].values
    capacity_norm = capacity / capacity[0]

    # =============================
    # CREATE SEQUENCES
    # =============================
    X_seq, y_seq = [], []

    for i in range(len(capacity_norm) - window):
        X_seq.append(capacity_norm[i:i+window])
        y_seq.append(capacity_norm[i+window])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

    # =============================
    # MODEL
    # =============================
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(window,1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_seq, y_seq, epochs=20, verbose=0)

    # =============================
    # FUTURE PREDICTION (ITERATIVE)
    # =============================
    last_sequence = capacity_norm[-window:].tolist()
    future_predictions = []

    for _ in range(future_max - len(capacity_norm)):

        seq_input = np.array(last_sequence[-window:])
        seq_input = seq_input.reshape((1, window, 1))

        pred = model.predict(seq_input, verbose=0)[0][0]

        future_predictions.append(pred)
        last_sequence.append(pred)

    # =============================
    # COMBINE
    # =============================
    pred_full = np.concatenate([capacity_norm, future_predictions])
    full_cycles = np.arange(1, len(pred_full) + 1)

    # =============================
    # FIND RUL (PRINT ONLY)
    # =============================
    rul = None
    for i, val in enumerate(pred_full):
        if val <= threshold:
            rul = full_cycles[i]
            break

    print(f"{name} | LSTM RUL:", rul)

    # =============================
    # 📊 PLOT (NO RUL LINE)
    # =============================
    plt.figure(figsize=(8,5))

    plt.plot(cycle, capacity_norm, label='Actual')
    plt.plot(full_cycles, pred_full, label='LSTM Prediction')

    plt.axhline(y=threshold, linestyle='--', label='80% Threshold')

    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.title(f"LSTM RUL Prediction (300 cycles) - {name}")
    plt.legend()
    plt.grid()

    plt.show()

print("\nLSTM RUL prediction completed!")
