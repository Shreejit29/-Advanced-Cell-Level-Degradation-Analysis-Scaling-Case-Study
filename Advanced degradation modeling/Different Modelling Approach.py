import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

# -----------------------------
# PATH
# -----------------------------
data_folder = "data/processed/"
files = glob(os.path.join(data_folder, "*.xlsx"))

# -----------------------------
# FUNCTION: RMSE
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# =============================
# LOOP THROUGH DATASETS
# =============================
for file in files:

    print(f"\nProcessing: {file}")

    df = pd.read_excel(file)
    name = os.path.basename(file)

    # -----------------------------
    # DATA PREPARATION
    # -----------------------------
    X = df[['Cycle_Number']].values
    y = df['Discharge_Capacity'].values

    # Normalize
    y_norm = y / y[0]

    # -----------------------------
    # LINEAR REGRESSION
    # -----------------------------
    lin_model = LinearRegression()
    lin_model.fit(X, y_norm)
    y_pred_lin = lin_model.predict(X)

    # -----------------------------
    # POLYNOMIAL REGRESSION
    # -----------------------------
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    poly_model = LinearRegression()
    poly_model.fit(X_poly, y_norm)
    y_pred_poly = poly_model.predict(X_poly)

    # -----------------------------
    # SEQUENCE DATA (RNN/LSTM)
    # -----------------------------
    window = 5
    data = y_norm

    X_seq, y_seq = [], []

    for i in range(len(data) - window):
        X_seq.append(data[i:i+window])
        y_seq.append(data[i+window])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

    # -----------------------------
    # RNN MODEL
    # -----------------------------
    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(50, activation='relu', input_shape=(window, 1)))
    model_rnn.add(Dense(1))

    model_rnn.compile(optimizer='adam', loss='mse')
    model_rnn.fit(X_seq, y_seq, epochs=20, verbose=0)

    y_pred_rnn = model_rnn.predict(X_seq).flatten()

    # -----------------------------
    # LSTM MODEL
    # -----------------------------
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(window, 1)))
    model_lstm.add(Dense(1))

    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_seq, y_seq, epochs=20, verbose=0)

    y_pred_lstm = model_lstm.predict(X_seq).flatten()

    # -----------------------------
    # PRINT ERRORS
    # -----------------------------
    print("Linear RMSE:", rmse(y_norm, y_pred_lin))
    print("Polynomial RMSE:", rmse(y_norm, y_pred_poly))
    print("RNN RMSE:", rmse(y_seq, y_pred_rnn))
    print("LSTM RMSE:", rmse(y_seq, y_pred_lstm))

    # =============================
    # 📊 SEPARATE PLOTS
    # =============================

    # Linear
    plt.figure()
    plt.plot(X, y_norm, 'o', label='Actual')
    plt.plot(X, y_pred_lin, '-', label='Linear')
    plt.title(f"{name} - Linear Regression")
    plt.legend()
    plt.grid()
    plt.show()

    # Polynomial
    plt.figure()
    plt.plot(X, y_norm, 'o', label='Actual')
    plt.plot(X, y_pred_poly, '-', label='Polynomial')
    plt.title(f"{name} - Polynomial Regression")
    plt.legend()
    plt.grid()
    plt.show()

    # RNN vs LSTM
    plt.figure()
    plt.plot(y_seq, label='Actual')
    plt.plot(y_pred_rnn, label='RNN')
    plt.plot(y_pred_lstm, label='LSTM')
    plt.title(f"{name} - RNN vs LSTM")
    plt.legend()
    plt.grid()
    plt.show()

    # =============================
    # 📊 COMBINED PLOT
    # =============================
    plt.figure()

    plt.plot(X, y_norm, 'o', label='Actual')
    plt.plot(X, y_pred_lin, '-', label='Linear')
    plt.plot(X, y_pred_poly, '--', label='Polynomial')

    plt.plot(range(window, len(y_norm)), y_pred_rnn, '-.', label='RNN')
    plt.plot(range(window, len(y_norm)), y_pred_lstm, ':', label='LSTM')

    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.title(f"{name} - Model Comparison")
    plt.legend()
    plt.grid()

    plt.show()

print("\nAll datasets processed successfully!")
