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

# PATHS 

data_folder = "data/processed/"

files = glob(os.path.join(data_folder, "*.xlsx"))

print("Files found:", files)

# FUNCTION: RMSE
-
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# FUNCTION: CREATE SEQUENCES

def create_seq(data, window):
    X_seq, y_seq = [], []
    for i in range(len(data) - window):
        X_seq.append(data[i:i+window])
        y_seq.append(data[i+window])
    return np.array(X_seq), np.array(y_seq)


# MAIN LOOP

for file in files:

    print(f"\nProcessing: {file}")

    df = pd.read_excel(file)
    name = os.path.basename(file).replace(".xlsx", "")

   
    # DATA PREP

    X = df[['Cycle_Number']].values
    y = df['Discharge_Capacity'].values

    # Normalize
    y_norm = y / y[0]

    
    # TIME-BASED SPLIT

    split_ratio = 0.7
    split_index = int(len(X) * split_ratio)

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y_norm[:split_index], y_norm[split_index:]

 
    #  LINEAR REGRESSION
    
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)

  
    #  POLYNOMIAL REGRESSION
    
    poly = PolynomialFeatures(degree=2)

    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_poly_train, y_train)
    y_pred_poly = poly_model.predict(X_poly_test)

   
    #  SEQUENCE PREP
    
    window = 5

    train_data = y_norm[:split_index]
    test_data = y_norm[split_index - window:]

    X_train_seq, y_train_seq = create_seq(train_data, window)
    X_test_seq, y_test_seq = create_seq(test_data, window)

    X_train_seq = X_train_seq.reshape((-1, window, 1))
    X_test_seq = X_test_seq.reshape((-1, window, 1))

    
    #  RNN MODEL
    
    model_rnn = Sequential()
    model_rnn.add(SimpleRNN(50, activation='relu', input_shape=(window, 1)))
    model_rnn.add(Dense(1))

    model_rnn.compile(optimizer='adam', loss='mse')
    model_rnn.fit(X_train_seq, y_train_seq, epochs=20, verbose=0)

    y_pred_rnn = model_rnn.predict(X_test_seq).flatten()

    
    #  LSTM MODEL
    
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(window, 1)))
    model_lstm.add(Dense(1))

    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_train_seq, y_train_seq, epochs=20, verbose=0)

    y_pred_lstm = model_lstm.predict(X_test_seq).flatten()

  
    #  RMSE
  
    print("Linear RMSE:", rmse(y_test, y_pred_lin))
    print("Polynomial RMSE:", rmse(y_test, y_pred_poly))
    print("RNN RMSE:", rmse(y_test_seq, y_pred_rnn))
    print("LSTM RMSE:", rmse(y_test_seq, y_pred_lstm))

    
    #  VISUALIZATION 

    plt.figure(figsize=(8,5))
    plt.plot(X_test, y_test, 'o', label='Actual')
    plt.plot(X_test, y_pred_lin, '-', label='Linear')
    plt.plot(X_test, y_pred_poly, '--', label='Polynomial')
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.title(f"{name} - Regression Models")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(y_test_seq, label='Actual')
    plt.plot(y_pred_rnn, label='RNN')
    plt.plot(y_pred_lstm, label='LSTM')
    plt.title(f"{name} - Sequence Models")
    plt.legend()
    plt.grid()
    plt.show()




print("\nAll models processed successfully!")
