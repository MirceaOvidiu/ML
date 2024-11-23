import wfdb as wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest

# Load the ECG record
record_path = "C:\\Anul III\\TIA\\datasets\\mit-bih-arrhythmia-database-1.0.0\\mit-bih-arrhythmia-database-1.0.0\\100"
record = wfdb.rdrecord(record_path)
signal = record.p_signal

# Preprocess the data
scaler = StandardScaler()
signal_scaled = scaler.fit_transform(signal)

# Fit the Isolation Forest model from pyod
iso_forest = IForest(contamination=0.1, random_state=42)
iso_forest.fit(signal_scaled)

# Predict anomalies
anomalies = iso_forest.predict(signal_scaled)

# Convert predictions to boolean (1 for anomaly, 0 for normal)
anomalies = np.where(anomalies == 1, 1, 0)

# Plot the results
plt.figure(figsize=(15, 5))
plt.plot(signal_scaled, label='ECG Signal')
plt.scatter(np.arange(len(signal_scaled)), signal_scaled[:, 0], c=anomalies, cmap='coolwarm', label='Anomalies')
plt.title('ECG Signal with Anomalies Detected by Isolation Forest')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()