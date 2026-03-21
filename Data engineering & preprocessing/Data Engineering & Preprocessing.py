import pandas as pd
import numpy as np
import os
from glob import glob

# -----------------------------
# PATH CONFIG
# -----------------------------
input_folder = "data/raw/"
output_folder = "data/Processed/"

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# TIME CONVERSION FUNCTION
# -----------------------------
def convert_time(x):
    try:
        if hasattr(x, 'hour'):
            return x.hour + x.minute/60 + x.second/3600

        x = str(x)

        if '-' in x:
            return pd.to_timedelta(x).total_seconds() / 3600

        dt = pd.to_datetime(x, format='%I:%M:%S %p')
        return dt.hour + dt.minute/60 + dt.second/3600

    except:
        return np.nan


# -----------------------------
# PROCESS ONE FILE
# -----------------------------
def process_file(file_path):

    df = pd.read_excel(file_path)

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    df = df.fillna(method='ffill').fillna(method='bfill')

    # -----------------------------
    # TIME PROCESSING
    # -----------------------------
    df['Time_Hours'] = df['Test_Time(s)'].apply(convert_time)
    df['Time_Hours'] = df['Time_Hours'].abs()

    # -----------------------------
    # REMOVE ANOMALIES
    # -----------------------------
    df = df[
        (df['Voltage(V)'] > 0) & (df['Voltage(V)'] < 5) &
        (df['Capacity(Ah)'] >= 0)
    ]

    # -----------------------------
    # CYCLE DURATION
    # -----------------------------
    cycle_time = df.groupby('Cycle_Index')['Time_Hours'].agg(['min', 'max'])
    cycle_time['Cycle_Duration'] = cycle_time['max'] - cycle_time['min']

    df = df.merge(cycle_time['Cycle_Duration'], on='Cycle_Index', how='left')

    # -----------------------------
    # CLEAN DATA
    # -----------------------------
    df = df.drop(['Test_Time(s)', 'Date_Time'], axis=1, errors='ignore')

    df = df[
        ['Cycle_Index', 'Time_Hours', 'Cycle_Duration',
         'Voltage(V)', 'Current(A)', 'Capacity(Ah)', 'Energy(Wh)']
    ]

    df = df[df['Current(A)'] != 0].reset_index(drop=True)

    # -----------------------------
    # AVERAGE CURRENT
    # -----------------------------
    charge_current = df[df['Current(A)'] > 0].groupby('Cycle_Index')['Current(A)'].mean()
    discharge_current = df[df['Current(A)'] < 0].groupby('Cycle_Index')['Current(A)'].mean()

    df = df.merge(charge_current.rename('Avg_Charge_Current'), on='Cycle_Index', how='left')
    df = df.merge(discharge_current.rename('Avg_Discharge_Current'), on='Cycle_Index', how='left')

    # -----------------------------
    # PHASE DETECTION
    # -----------------------------
    df['Current_Sign'] = df['Current(A)'] > 0
    df['Sign_Change'] = df['Current_Sign'] != df['Current_Sign'].shift(1)

    end_of_phases = []

    for i in range(1, len(df)):
        if df.iloc[i]['Sign_Change']:
            end_of_phases.append(i - 1)

    if len(df) > 0 and df.iloc[-1]['Current(A)'] < 0:
        end_of_phases.append(len(df) - 1)

    final_dataset = df.iloc[end_of_phases].copy().reset_index(drop=True)

    final_dataset = final_dataset.drop(['Current_Sign', 'Sign_Change'], axis=1)

    # -----------------------------
    # SPLIT CHARGE / DISCHARGE
    # -----------------------------
    final_dataset['Charge_Capacity'] = np.where(final_dataset['Current(A)'] > 0,
                                                final_dataset['Capacity(Ah)'], 0)

    final_dataset['Discharge_Capacity'] = np.where(final_dataset['Current(A)'] < 0,
                                                   final_dataset['Capacity(Ah)'], 0)

    final_dataset['Charge_Energy'] = np.where(final_dataset['Current(A)'] > 0,
                                              final_dataset['Energy(Wh)'], 0)

    final_dataset['Discharge_Energy'] = np.where(final_dataset['Current(A)'] < 0,
                                                 final_dataset['Energy(Wh)'], 0)

    # Align discharge with charge
    final_dataset['Discharge_Capacity'] = final_dataset['Discharge_Capacity'].shift(-1).fillna(0)
    final_dataset['Discharge_Energy'] = final_dataset['Discharge_Energy'].shift(-1).fillna(0)

    # Keep valid cycles
    final_dataset = final_dataset[
        (final_dataset['Charge_Capacity'] > 0) &
        (final_dataset['Discharge_Capacity'] > 0)
    ].reset_index(drop=True)

    # -----------------------------
    # CYCLE NUMBER
    # -----------------------------
    final_dataset.insert(0, 'Cycle_Number', range(1, len(final_dataset) + 1))

    # -----------------------------
    # EFFICIENCIES
    # -----------------------------
    final_dataset['Coulombic_Efficiency'] = (
        final_dataset['Discharge_Capacity'] / final_dataset['Charge_Capacity']
    )

    final_dataset['Energy_Efficiency'] = (
        final_dataset['Discharge_Energy'] / final_dataset['Charge_Energy']
    )

    # -----------------------------
    # ENERGY THROUGHPUT
    # -----------------------------
    final_dataset['Energy_Throughput'] = (
        final_dataset['Charge_Energy'] + final_dataset['Discharge_Energy']
    )

    # -----------------------------
    # C-RATES
    # -----------------------------
    nominal_capacity = final_dataset['Charge_Capacity'].iloc[0]

    final_dataset['Charge_C_Rate'] = final_dataset['Avg_Charge_Current'] / nominal_capacity
    final_dataset['Discharge_C_Rate'] = abs(final_dataset['Avg_Discharge_Current']) / nominal_capacity

    # -----------------------------
    # FINAL CLEANING
    # -----------------------------
    final_dataset = final_dataset.fillna(method='ffill').fillna(method='bfill')

    final_dataset = final_dataset[
        (final_dataset['Coulombic_Efficiency'] <= 1.05) &
        (final_dataset['Coulombic_Efficiency'] > 0)
    ]

    return final_dataset, base_name


# -----------------------------
# PROCESS ALL FILES
# -----------------------------
for file_path in glob(os.path.join(input_folder, "*.xlsx")):

    final_dataset, name = process_file(file_path)

    output_file = os.path.join(output_folder, f"{name}_processed.xlsx")

    # Save to Excel
    final_dataset.to_excel(output_file, index=False)

    print(f"Processed and saved: {output_file}")

print("All files processed successfully!")
