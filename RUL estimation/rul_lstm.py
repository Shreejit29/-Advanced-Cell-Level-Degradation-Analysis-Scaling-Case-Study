import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =============================
# PARAMETERS
# =============================
window = 5
threshold = 0.8
max_cycle_limit = 100

# =============================
# PATHS
# =============================
data_folder = "data/processed/"


files = glob(os.path.join(data_folder, "*.xlsx"))

print("Files found:", files)

# =============================
# LOOP
# =============================
for file in files:

    print("\n" + "="*50)
    print(f"Processing: {file}")

    df = pd.read_excel(file)
    name = os.path.basename(file).replace(".xlsx", "")

    # -----------------------------
    # DATA
    # -----------------------------
    cycle = df['Cycle_Number'].values
    capacity = df['Discharge_Capacity'].values
    capacity_norm = capacity / capacity[0]

    # =============================
    # ACTUAL FAILURE
    # =============================
    try:
        actual_idx = np.where(capacity_norm <= threshold)[0][0]
        actual_failure = cycle[actual_idx]
    except:
        actual_failure = None

    print(f"Actual Failure Cycle: {actual_failure}")

    # =============================
    # 🔥 EARLY TRAINING
    # =============================
    if actual_failure is not None:
        cutoff_cycle = int(0.6 * actual_failure)
    else:
        cutoff_cycle = int(0.6 * len(cycle))

    mask = cycle <= cutoff_cycle
    cap_early = capacity_norm[mask]
    cycle_early = cycle[mask]

    current_cycle = cycle_early[-1]

    print(f"Training up to cycle: {current_cycle}")

    # =============================
    # CREATE SEQUENCES
    # =============================
    X_seq, y_seq = [], []

    for i in range(len(cap_early) - window):
        X_seq.append(cap_early[i:i+window])
        y_seq.append(cap_early[i+window])

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
    model.fit(X_seq, y_seq, epochs=30, verbose=0)

    # =============================
    # FUTURE PREDICTION
    # =============================
    last_sequence = cap_early[-window:].tolist()
    future_predictions = []

    future_steps = max_cycle_limit - current_cycle

    for _ in range(future_steps):

        seq_input = np.array(last_sequence[-window:])
        seq_input = seq_input.reshape((1, window, 1))

        pred = model.predict(seq_input, verbose=0)[0][0]

        future_predictions.append(pred)
        last_sequence.append(pred)

    # =============================
    # COMBINE
    # =============================
    pred_full = np.concatenate([cap_early, future_predictions])
    full_cycles = np.arange(1, len(pred_full) + 1)

    # =============================
    # FIND FAILURE + RUL
    # =============================
    predicted_failure = None
    rul = None
    error = None

    for i, val in enumerate(pred_full):
        if val <= threshold:
            predicted_failure = full_cycles[i]
            rul = predicted_failure - current_cycle

            if actual_failure is not None:
                error = abs(predicted_failure - actual_failure)
            break

    print(f"\n{name} | LSTM")
    print(f"Predicted Failure Cycle: {predicted_failure}")
    print(f"RUL: {rul}")
    print(f"Error: {error}")

    # =============================
    # 📊 PLOT
    # =============================
    plt.figure(figsize=(8,5))

    plt.plot(cycle, capacity_norm, color='black', label='Actual')
    plt.plot(full_cycles, pred_full, color='blue', label='LSTM Prediction')

    plt.axhline(y=threshold, color='red', linestyle='--', label='80% Threshold')

    if predicted_failure is not None:
        plt.axvline(x=predicted_failure, linestyle='--', label='Predicted')

    if actual_failure is not None:
        plt.axvline(x=actual_failure, color='red', linestyle='-', label='Actual')

    plt.axvline(x=current_cycle, color='purple', linestyle=':', label='Training Cutoff')

    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.title(f"LSTM Early Prediction - {name}")
    plt.legend()
    plt.grid()


    plt.show()

print("\n✅ LSTM early prediction complete!")
