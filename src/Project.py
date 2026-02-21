import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import time
from tensorflow.keras import layers, models

# --- 1. SETTINGS & PREVENT LATEX ERROR ---
plt.rcParams.update({'text.usetex': False})
WINDOW_SIZE = 5
HEALTHY_RUL_THRESHOLD = 125 # Cycles before failure where degradation typically begins

# --- 2. DOWNLOAD & LOAD ---
path = kagglehub.dataset_download("palbha/cmapss-jet-engine-simulated-data")
print("Path to dataset files:", path)

columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's_{i}' for i in range(1, 22)]
df = pd.read_csv(f"{path}/train_FD001.txt", sep='\s+', header=None, names=columns)
df_test = pd.read_csv(f"{path}/test_FD001.txt", sep='\s+', header=None, names=columns)

# --- 3. CALCULATE RUL FOR TRAINING DATA ---
# Since engines in the train set run to failure, RUL is (Max Cycle - Current Cycle)
max_cycles = df.groupby('id')['cycle'].max().reset_index()
max_cycles.columns = ['id', 'max_cycle']
df = df.merge(max_cycles, on=['id'], how='left')
df['RUL'] = df['max_cycle'] - df['cycle']
print("Training Data Sample:")
print(df.head())
df.drop('max_cycle', axis=1, inplace=True)

# --- 4. CLEANING ---
# Drop constant sensors and settings
drop_cols = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19', 'setting1', 'setting2', 'setting3']
df.drop(labels=drop_cols, axis=1, inplace=True)
df_test.drop(labels=drop_cols, axis=1, inplace=True)

sensor_cols = [c for c in df.columns if c.startswith('s_')]

# --- 5. SCALING ---
# Fit the scaler ONLY on the healthy data (RUL > 125) to prevent degraded data from skewing it
healthy_mask = df['RUL'] > HEALTHY_RUL_THRESHOLD
scaler = MinMaxScaler()
scaler.fit(df[healthy_mask][sensor_cols])

df[sensor_cols] = scaler.transform(df[sensor_cols])
df_test[sensor_cols] = scaler.transform(df_test[sensor_cols])

# --- 6. SEQUENCING FUNCTION (Upgraded) ---
# Now returns sequences AND their corresponding exact cycle and RUL
def gen_sequences_and_labels(data, window):
    sequences, cycles, ruls = [], [], []
    for unit_id in data['id'].unique():
        unit_data = data[data['id'] == unit_id]
        sensor_vals = unit_data[sensor_cols].values
        cycle_vals = unit_data['cycle'].values
        rul_vals = unit_data['RUL'].values if 'RUL' in unit_data.columns else np.zeros(len(unit_data))
        
        if len(unit_data) >= window:
            for i in range(len(unit_data) - window):
                sequences.append(sensor_vals[i:i+window])
                # Track the cycle and RUL at the END of the window
                cycles.append(cycle_vals[i+window])
                ruls.append(rul_vals[i+window])
                
    return np.array(sequences), np.array(cycles), np.array(ruls)

# Generate sequences for the entire training set
X_train_full, cycles_train, ruls_train = gen_sequences_and_labels(df, WINDOW_SIZE)

# Filter for purely healthy sequences to train the Autoencoder
healthy_idx = ruls_train > HEALTHY_RUL_THRESHOLD
X_train_healthy = X_train_full[healthy_idx]

# --- 7. LSTM AUTOENCODER ---
model = models.Sequential([
    layers.Input(shape=(WINDOW_SIZE, len(sensor_cols))),
    layers.LSTM(32, activation='relu', return_sequences=False),
    layers.RepeatVector(WINDOW_SIZE),
    layers.LSTM(32, activation='relu', return_sequences=True),
    layers.TimeDistributed(layers.Dense(len(sensor_cols)))
])

model.compile(optimizer='adam', loss='mae')
print(f"Training on {len(X_train_healthy)} healthy sequences...")
model.fit(X_train_healthy, X_train_healthy, epochs=15, batch_size=64, validation_split=0.1, verbose=1)

# --- 8. DATA-DRIVEN THRESHOLD CALCULATION ---
# Get reconstruction error for all training data
train_predictions = model.predict(X_train_full)
train_mae = np.mean(np.abs(train_predictions - X_train_full), axis=(1, 2))

# The threshold is the 95th percentile of error while the engine is STILL HEALTHY
healthy_mae = train_mae[healthy_idx]
smart_threshold = np.percentile(healthy_mae, 90)
print(f"Data-Driven Failure Threshold: {smart_threshold:.4f}")

# --- 9. VISUALIZATION (UNIT 100) ---
unit_id = 1
unit_mask = df['id'] == unit_id
X_unit, cycles_unit, ruls_unit = gen_sequences_and_labels(df[unit_mask], WINDOW_SIZE)

unit_preds = model.predict(X_unit)
unit_mae = np.mean(np.abs(unit_preds - X_unit), axis=(1, 2))

# Find the exact cycle where RUL hits our 125 threshold
fault_cycle = cycles_unit[ruls_unit <= HEALTHY_RUL_THRESHOLD][0] if any(ruls_unit <= HEALTHY_RUL_THRESHOLD) else cycles_unit[-1]

plt.figure(figsize=(12, 6))
plt.plot(cycles_unit, unit_mae, label='Anomaly Score (Reconstruction Error)', color='blue')
plt.axhline(y=smart_threshold, color='red', linestyle='-', label=f'Smart Threshold ({smart_threshold:.4f})')
plt.axvline(x=fault_cycle, color='orange', linestyle='--', label=f'Actual Fault Point (Cycle {fault_cycle})')

plt.title(f'Engine {unit_id} Degradation Tracking (Training Set)')
plt.xlabel('Operating Cycle')
plt.ylabel('MAE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show(block=False) # Prevent visualization from freezing the script before testing

# --- 10. TEST SET EVALUATION ---
print("\n--- Starting Test Set Evaluation ---")

# Load the True RUL values for the FD001 test set
true_rul_file = f"{path}/RUL_FD001.txt"
true_rul_df = pd.read_csv(true_rul_file, header=None, names=['True_RUL'])
true_rul_df['id'] = true_rul_df.index + 1  # IDs start at 1

# Find the last recorded cycle for each engine in the test set
test_max_cycles = df_test.groupby('id')['cycle'].max().reset_index()
test_max_cycles.columns = ['id', 'last_test_cycle']

# Merge to get the full picture
eval_df = test_max_cycles.merge(true_rul_df, on='id')
eval_df['total_life'] = eval_df['last_test_cycle'] + eval_df['True_RUL']
eval_df['actual_fault_cycle'] = eval_df['total_life'] - HEALTHY_RUL_THRESHOLD

predicted_faults = []

for u_id in eval_df['id']:
    unit_mask = df_test['id'] == u_id
    unit_data = df_test[unit_mask]
    
    # We can only predict if the test engine has more cycles than our WINDOW_SIZE
    if len(unit_data) >= WINDOW_SIZE:
        X_test_unit, cycles_test_unit, _ = gen_sequences_and_labels(unit_data, WINDOW_SIZE)
        
        # Get Anomaly Scores for the test unit
        unit_test_preds = model.predict(X_test_unit, verbose=0)
        unit_test_mae = np.mean(np.abs(unit_test_preds - X_test_unit), axis=(1, 2))
        
        # Find where the anomaly score crosses the threshold
        fault_indices = np.where(unit_test_mae > smart_threshold)[0]
        
        if len(fault_indices) > 0:
            # The first time it crosses the threshold is our predicted fault point
            first_fault_idx = fault_indices[0]
            predicted_fault_cycle = cycles_test_unit[first_fault_idx]
        else:
            # Model never detected a fault in the available test window
            predicted_fault_cycle = np.nan
    else:
        predicted_fault_cycle = np.nan
        
    predicted_faults.append(predicted_fault_cycle)

eval_df['predicted_fault_cycle'] = predicted_faults

# Compute Accuracy Metrics
# Drop engines where the model didn't have enough data or didn't trigger
valid_eval = eval_df.dropna(subset=['predicted_fault_cycle']).copy()

# Calculate the difference (Prediction - Actual)
valid_eval['detection_error'] = valid_eval['predicted_fault_cycle'] - valid_eval['actual_fault_cycle']

# MAE in Cycles
mean_error_cycles = np.mean(np.abs(valid_eval['detection_error']))

# Categorize the detections
early_alarms = len(valid_eval[valid_eval['detection_error'] < -10]) # Triggered more than 10 cycles early
late_detections = len(valid_eval[valid_eval['detection_error'] > 10]) # Triggered more than 10 cycles late
on_time = len(valid_eval[(valid_eval['detection_error'] >= -10) & (valid_eval['detection_error'] <= 10)])

print(f"--- TEST SET RESULTS ---")
print(f"Total Engines Evaluated: {len(valid_eval)} (out of {len(eval_df)} total test engines)")
print(f"Mean Detection Error: {mean_error_cycles:.2f} cycles")
print(f"On-Time Detections (within +/- 10 cycles): {on_time}")
print(f"Early False Alarms: {early_alarms}")
print(f"Late Detections: {late_detections}")

plt.show() # Keep the plot open at the very end