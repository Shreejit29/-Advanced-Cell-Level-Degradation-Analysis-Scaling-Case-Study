import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans



# PATHS 

data_folder = "data/processed/"

files = glob(os.path.join(data_folder, "*.xlsx"))

print("Files found:", files)



# LOOP THROUGH FILES

for file in files:

    print(f"\nProcessing: {file}")

    df = pd.read_excel(file)
    name = os.path.basename(file).replace(".xlsx", "")


    # FEATURES 
   
    features = df[[
        'Discharge_Capacity',
        'Coulombic_Efficiency',
        'Energy_Efficiency'
    ]].dropna()

    df = df.loc[features.index]

    # NORMALIZATION

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

   
    #  ISOLATION FOREST
   
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['ISO'] = iso.fit_predict(X)
    print("ISO anomalies:", np.sum(df['ISO'] == -1))


    #  LOF
  
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    df['LOF'] = lof.fit_predict(X)
    print("LOF anomalies:", np.sum(df['LOF'] == -1))


    #  MAHALANOBIS (STABLE)

    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    # Regularization for stability
    cov += np.eye(cov.shape[0]) * 1e-6

    inv_cov = np.linalg.inv(cov)

    def mahalanobis(x):
        return np.sqrt((x - mean).T @ inv_cov @ (x - mean))

    distances = np.array([mahalanobis(x) for x in X])

    threshold = np.percentile(distances, 95)
    df['Mahalanobis'] = np.where(distances > threshold, -1, 1)
    print("Mahalanobis anomalies:", np.sum(df['Mahalanobis'] == -1))

    
    #  K-MEANS

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    centers = kmeans.cluster_centers_
    distances_k = np.linalg.norm(X - centers[clusters], axis=1)

    threshold_k = np.percentile(distances_k, 95)
    df['KMeans'] = np.where(distances_k > threshold_k, -1, 1)
    print("KMeans anomalies:", np.sum(df['KMeans'] == -1))

    
    # ENSEMBLE
   
    df['Final_Anomaly'] = (
        (df['ISO'] == -1).astype(int) +
        (df['LOF'] == -1).astype(int) +
        (df['Mahalanobis'] == -1).astype(int) +
        (df['KMeans'] == -1).astype(int)
    )

    df['Final_Anomaly'] = np.where(df['Final_Anomaly'] >= 2, -1, 1)
    print("Final anomalies:", np.sum(df['Final_Anomaly'] == -1))

    
    #  PLOT 
  
    def plot_anomaly(label_column, title):

        plt.figure(figsize=(8,5))

        normal = df[df[label_column] == 1]
        outliers = df[df[label_column] == -1]

        plt.scatter(normal['Cycle_Number'], normal['Discharge_Capacity'], label='Healthy')
        plt.scatter(outliers['Cycle_Number'], outliers['Discharge_Capacity'], color='red', label='Outlier')

        plt.xlabel("Cycle Number")
        plt.ylabel("Discharge Capacity")
        plt.title(title)
        plt.legend()
        plt.grid()

        plt.show()

    =
    #  PLOTS
    
    plot_anomaly('ISO', f"{name} - Isolation Forest")
    plot_anomaly('LOF', f"{name} - LOF")
    plot_anomaly('Mahalanobis', f"{name} - Mahalanobis")
    plot_anomaly('KMeans', f"{name} - KMeans")
    plot_anomaly('Final_Anomaly', f"{name} - Ensemble")

   


print("\nAnomaly detection completed successfully!")
