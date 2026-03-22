Advanced-Cell-Level-Degradation-Analysis-Scaling-Case-Study
This project provides a comprehensive analysis of battery capacity degradation and remaining useful life (RUL) using experimental cycling data. It processes raw battery test data, computes health metrics, detects anomalous cycles, and applies multiple modeling approaches (statistical and machine learning) to predict end-of-life. Key steps include cleaning and structuring the data, deriving features like Coulombic/Energy Efficiency and a Combined Efficiency Factor (CEF), applying anomaly detection algorithms, and forecasting RUL with extrapolation, linear regression, and LSTM models. The goal is to identify the cycle at which capacity drops below 80% of initial value (the defined end-of-life threshold) and to flag early signs of degradation.
•	Data Preprocessing: Clean raw data and aggregate per cycle. 
•	Efficiency Metrics: Compute per-cycle Coulombic Efficiency, Energy Efficiency, and the Combined Efficiency Factor (CEF). 
•	Anomaly Detection: Use statistical and ML methods (Isolation Forest, LOF, Mahalanobis, K-Means) to flag abnormal cycles.
•	Advanced Degradation Modeling: Analyze battery degradation behavior using multiple approaches: Capacity fade analysis (cycle-by-cycle degradation trends), Different modelling approaches (regression-based fitting), Differential voltage (dV/dQ) analysis for electrochemical insights and Coulombic efficiency trend analysis for performance degradation
•	RUL Modeling: Predict remaining life through polynomial extrapolation, linear regression with CEF, and LSTM-based forecasting. 
These components form an end-to-end pipeline for battery health evaluation and RUL prediction.
Folder Structure
•	data/raw/ – Contains the raw battery test data files (Excel format). Each file holds the original time-series measurements for charging and discharging cycles. 
•	data/Processed/ – Stores the processed output data files (Excel). After preprocessing, each file has one row per cycle with computed features (capacities, energies, efficiencies). 
•	Python Scripts (root or scripts/ directory): The main analysis scripts include: 
•	Data Engineering & Preprocessing.py – Cleans and restructures raw data into cycle-level format.
In Advanced degradation modeling:
•	Capacity fade analysis.py – Performs capacity fade analysis using empirical models such as exponential and square-root degradation.
•	Differential voltage dv dq.py – Computes differential voltage (dV/dQ) curves to analyze electrode-level degradation signatures.
•	Different Modelling Approach.py – Applies statistical regression models (linear and polynomial) and implements sequence-based modeling using LSTM to capture temporal degradation behavior.
•	Coulombic efficiency trend.py - This script analyzes the trend of Coulombic Efficiency across cycles
•	Anomaly detection.py – Identifies anomalous cycles using multiple unsupervised methods.
•	rul_extrapolation.py – Fits polynomial models on early-cycle data to estimate RUL.
•	rul_ml_cef.py – Performs linear regression with Cycle_Number and CEF features to predict future capacity.
•	rul_lstm.py – Trains an LSTM network on capacity sequences for RUL forecasting.
•	Plots & Visualisations– Stores generated visualizations and analysis outputs.
This folder contains all plots generated during model execution (if saved manually). These visualizations help in understanding battery degradation behavior, anomaly detection, and RUL prediction.
•	Includes:
•	Advanced Degradation Modeling Plots:
•	Capacity vs. cycle degradation curves
•	Model fitting comparisons (linear, polynomial, etc.)
•	dV/dQ curves showing electrochemical changes
•	Coulombic efficiency trend plots
•	Anomaly Detection Plots:
•	Scatter plots highlighting healthy vs. outlier cycles
•	RUL Estimation Plots:
•	Capacity extrapolation curves with failure threshold
•	Predicted vs. actual capacity (Linear model & LSTM)
•	Feature Importance Plots:
•	SHAP analysis plots showing feature impact (Cycle Number, CEF)
•	Other Files: Project documentation or reports (e.g. analysis write-ups) may be present in additional folders like docs/ or as standalone documents. 
•	Outputs: The scripts generate console outputs and plots unless data/Processed/ and data/raw/ manually saved. 
Installation Instructions
1.	Python Environment: Ensure Python 3.x is installed (Python 3.7 or higher recommended). Optionally create and activate a virtual environment. 
2.	Dependencies: Install required libraries. For example:
pip install pandas numpy scikit-learn matplotlib shap tensorflow
3.	Clone Project: Download or clone the repository to your local machine. 
4.	Data Folders: Verify that the folder structure exists. Create data/raw/ and data/Processed/ if they are not already present. 
5.	Prepare Raw Data: Place the raw Excel data files into the data/raw/ directory. 
6.	Run Scripts: Execute the Python scripts from the project root (instructions below). Ensure current working directory is the project root so that relative paths to data/ are correct. 
Dataset Format
•	Raw Data (Input): Excel (.xlsx) files in the data/raw/ folder. Each file contains time-series data for each charge/discharge cycle, with columns such as Test_Time(s), Voltage(V), Current(A), Capacity(Ah), etc. The data may include multiple steps (charge, rest, discharge) per cycle. 
•	Processed Data (Output): After running the preprocessing script, each Excel file in data/Processed/ has one row per cycle. Main columns include: 
	Cycle_Number: Sequential index of the cycle (starting at 1). 
	Charge_Capacity and Discharge_Capacity (in Ah). 
	Charge_Energy and Discharge_Energy (in Wh). 
	Coulombic_Efficiency (Discharge_Capacity / Charge_Capacity). 
	Energy_Efficiency (Discharge_Energy / Charge_Energy). 
	Combined_Efficiency_Factor (CEF): A derived metric combining CE and EE (calculated in downstream scripts) to capture overall efficiency. 
	Charge_C_Rate and Discharge_C_Rate: Charging/discharging current normalized by nominal capacity. 
	(Other columns such as cycle duration and aggregated currents may also be present.)
	These processed files serve as the input for anomaly detection and RUL modeling. 
Step-by-Step Instructions for Running Scripts
1.	Preprocess Data:
Run:
python "Data Engineering & Preprocessing.py"
This script reads all raw Excel files from data/raw/, fills missing values, computes time and energy metrics per cycle, and outputs cleaned cycle-level Excel files to data/Processed/ (one file per input). It prints messages confirming each processed file.
2.	Advanced Degradation Modeling:
(a) Capacity Fade Modeling (Empirical Models):
Run:
python "capacity fade analysis.py" 
This script performs cycle-by-cycle capacity degradation analysis. It tracks how battery capacity decreases over time and helps identify the overall degradation trend. 
(b) Differential Voltage Analysis (dV/dQ):
Run:
python "differential voltage dv dq.py"
This script computes differential voltage (dV/dQ) curves to analyze internal electrochemical changes. It helps detect electrode-level degradation and shifting voltage characteristics 
(c) Different Modelling Approach:
Run:
python "Different Modelling Approach.py"
This script applies multiple modeling techniques (e.g., regression-based approaches) to fit degradation behavior. It compares how different models capture battery aging patterns.
(d) Coulombic Efficiency Trend Analysis:
Run:
python "Coulombic efficiency trend.py"
This script analyzes the trend of Coulombic Efficiency across cycles. It helps identify efficiency loss, side reactions, and early degradation signals. 
3.	Anomaly Detection:
Run:
python "Anomaly detection.py"
This script loads each processed data file and selects features (Discharge_Capacity, Coulombic_Efficiency, Energy_Efficiency). It standardizes the data and applies four methods: Isolation Forest, Local Outlier Factor (LOF), Mahalanobis distance, and K-Means clustering. Each method labels cycles as normal or anomalous (with –1 for anomalies). The console output shows the number of anomalies found by each method. The script also generates scatter plots of cycle number vs. capacity, highlighting detected outliers (plots appear via plt.show()).
4.	Polynomial Extrapolation (RUL):
Run:
python rul_extrapolation.py
This script reads each processed data file and normalizes capacity (initial capacity = 1). It determines the cutoff cycle (60% of known cycles or 60% of actual failure cycle) and fits polynomial regression models of degree 1, 2, and 3 to the early-cycle data. It then extrapolates capacity until a maximum cycle (default 100) and finds where capacity falls to 80%. For each degree, it prints the predicted failure cycle, the remaining cycles (RUL), and the prediction error versus the actual failure. The script also plots the capacity curve and fitted polynomial predictions (degree 1 in blue, 2 in green, 3 in orange, actual data in black).
5.	Feature-Based Regression with CEF:
Run:
python rul_ml_cef.py
This script computes the Combined Efficiency Factor (CEF) for each cycle using Coulombic and Energy Efficiency values. It selects the first few cycles (e.g., train_cycles=5 by default) as training data and fits a linear regression model with features [Cycle_Number, CEF] to predict normalized capacity. It then predicts capacity for future cycles (assuming CEF decays exponentially) up to the max cycle. The script identifies the predicted failure cycle (where capacity ≤0.8) and computes RUL and error. It prints these results to the console. Additionally, it performs SHAP analysis on the linear model, displaying the mean absolute SHAP value for each feature (Cycle_Number, CEF), which indicates their importance in the model. A summary plot of feature impacts is also shown.
6.	Sequence Model (LSTM):
Run:
python rul_lstm.py
This script prepares the normalized capacity sequence and selects early-cycle data (up to 60% of the cycle range). It constructs overlapping sequences of a fixed window size (default 5) and trains an LSTM neural network to predict the next capacity value. After training (30 epochs by default), it feeds the final sequence to recursively forecast capacity until the maximum cycle. It then finds when the predicted capacity reaches the 0.8 threshold. The console output displays the predicted failure cycle and RUL (cycles remaining), along with the prediction error if the actual failure is known. A plot of actual vs. LSTM-predicted capacity over cycles is shown.
Note: The LSTM may require parameter tuning (window size, network size, epochs) and sufficient data for stable predictions. Use a machine with TensorFlow/GPU support if available for faster training.
Outputs Generated
•	Processed Data Files: After preprocessing, cleaned Excel files are saved in data/Processed/. Each file corresponds to an input raw file (e.g., Raw_dataset_1.xlsx). 
•	Console Reports: Each script prints results to the terminal. The anomaly detection script reports the count of outlier cycles per method. The RUL scripts report predicted failure cycles, remaining life (RUL), and any error compared to known failure. Also Advanced Degradation Modeling:
•	Capacity fade analysis prints degradation trends and fitted model parameters. 
•	Different modelling approaches report model coefficients and comparison insights. 
•	dV/dQ analysis provides information on peak behavior and voltage characteristics. 
•	Coulombic efficiency trend displays efficiency values and variation across cycles.
•	Visualizations: During execution, the scripts display plots all plots saved in Plots and Visualization Folder: 
•	Anomaly Detection: Scatter plots of each cycle’s capacity, with normal points and outliers highlighted. 
	Advanced Degradation Modeling:
•	Capacity Fade Analysis: Capacity vs. cycle plots showing degradation trends over time. 
•	Different Modelling Approach: Comparison plots of multiple fitted models (linear, polynomial, etc.) against actual capacity data. 
•	Differential Voltage (dV/dQ): dV/dQ curves highlighting peak shifts and electrochemical changes across cycles. 
•	Coulombic Efficiency Trend: Efficiency vs. cycle plots showing how Coulombic efficiency evolves with aging.
•	Extrapolation: Capacity vs. cycle plots showing actual data and polynomial fits (for degrees 1–3). 
•	LSTM: A plot of actual vs. predicted capacity over all cycles.
These plots are shown on screen via Matplotlib (plt.show()). (They can be saved manually if needed; the scripts do not automatically save image files.) 
•	SHAP Analysis: The rul_ml_cef.py script prints out the mean absolute SHAP values for each feature, showing their relative importance in the linear model. This helps interpret whether cycle number or CEF has more influence on the prediction. 
