import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from typing import *
from scipy.optimize import curve_fit

plt.style.use('tableau-colorblind10')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16
})

blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
orange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]


def failure_plot(anomaly_score: Any, threshold: Any, unit_no: int, savefig: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(anomaly_score, label='Reconstruction Error (Anomaly Score)')
    ax.axhline(y=threshold, linestyle='--', color=orange, label='Quiet Failure Threshold')
    try:
        alert_idx = np.where(anomaly_score > threshold)[0][0]
        plt.axvline(x=alert_idx, color=orange, alpha=0.5, label='Fault Detected')
    except:
        pass
    ax.set_title(f'Engine Unit {unit_no}: Quiet Failure Detection Over Time')
    ax.set_xlabel('Cycle (Excluding Window Offset)')
    ax.set_ylabel('MAE Error')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', top=True, right=True, left=True, bottom=True, which='both', direction='in')
    
    ax.legend()
    fig.tight_layout()

    if savefig:
        fig.savefig(savefig)

    return fig

def sensor_explore_plot(df, ids, values):
    unit_data = df[df['Unit'] == 1]
    plot_data = unit_data.melt(id_vars=ids, value_vars=values)

    # 2. Create a grid of plots
    g = sns.FacetGrid(plot_data, col="variable", col_wrap=4, sharey=False, aspect=1.5)
    g.map(sns.lineplot, "Cycle Time", "value")

    # Add a title and adjust layout
    g.figure.suptitle('Sensor Trends over Engine Life (Unit 1)', y=1.02, fontsize=16)

    return g

def gen_train_windows_multi(df, window_size, op_cols, sensor_cols):
    X, Y = [], []
    all_features = op_cols + sensor_cols
    
    for unit in df['Unit'].unique():
        # Get all relevant data for this unit
        unit_data = df[df['Unit'] == unit]
        
        # We only train on the 'Healthy' start (first 50 cycles)
        # Input (X) gets Ops + Sensors | Target (Y) gets Sensors Only
        input_data = unit_data[all_features].values[:50]
        target_data = unit_data[sensor_cols].values[:50]
        
        if len(input_data) >= window_size:
            for i in range(len(input_data) - window_size + 1):
                X.append(input_data[i:i+window_size])
                Y.append(target_data[i:i+window_size])
                
    return np.array(X), np.array(Y)

def gen_test_windows_multi(df, window_size, op_cols, sensor_cols):
    X, Y_actual, unit_ids = [], [], []
    all_features = op_cols + sensor_cols
    
    for unit in df['Unit'].unique():
        unit_data = df[df['Unit'] == unit]
        
        if len(unit_data) >= window_size:
            # X MUST have (Ops + Sensors) to match model Input
            X.append(unit_data[all_features].values[-window_size:])
            
            # Y_actual MUST have (Sensors Only) to match model Output
            Y_actual.append(unit_data[sensor_cols].values[-window_size:])
            
            unit_ids.append(unit)
            
    return np.array(X), np.array(Y_actual), unit_ids

def get_engine_history_multi(df, unit_id, window_size, op_cols, sensor_cols):
    unit_data = df[df['Unit'] == unit_id]
    all_features = op_cols + sensor_cols
    
    # Input for model (Needs 16 columns: Ops + Sensors)
    X_input = []
    # Actual sensors for comparison (Needs 14 columns: Sensors only)
    Y_actual = []
    
    # Convert to values
    input_values = unit_data[all_features].values
    target_values = unit_data[sensor_cols].values
    
    for i in range(len(input_values) - window_size + 1):
        X_input.append(input_values[i:i + window_size])
        Y_actual.append(target_values[i:i + window_size])
        
    return np.array(X_input), np.array(Y_actual)

def failure_detection(df, model, unit, threshold, wsize, ops, relevant_sensors):
    # 1. Get the sequences (Using your 'ops' and 'health_sensors' lists)
    X_input_seq, Y_actual_seq = get_engine_history_multi(
        df, 
        unit_id=unit, 
        window_size=wsize, 
        op_cols=ops, 
        sensor_cols=relevant_sensors
    )

    # 2. Predict (Model takes 16-col input, gives 14-col output)
    reconstructed = model.predict(X_input_seq)

    # 3. Calculate Anomaly Score
    # Compare the (samples, 30, 14) reconstruction to the (samples, 30, 14) actuals
    anomaly_score = np.mean(np.abs(Y_actual_seq - reconstructed), axis=(1, 2))

    # 4. Detection Logic
    try:
        # Adding +wsize gives you the actual 'Cycle' on the X-axis of the original data
        alert_cycle = np.where(anomaly_score > threshold)[0][0]
        print(f"Quiet Failure detected at Cycle: {alert_cycle}")
    except IndexError:
        print('No Failure detected above threshold.')
    
    return anomaly_score

def generate_audit_results(df, df_rul, model, wsize, ops, relevant_sensors):
    audit_results = []

    # 2. Loop through each test engine
    for i, unit_id in enumerate(df['Unit'].unique()):
        # Generate windows for this specific engine
        # (Ensure you use the same 'all_features' and 'relevant_sensors' as training)
        X_input, Y_actual = get_engine_history_multi(df, unit_id, wsize, ops, relevant_sensors)
        
        if len(X_input) > 0:
            # Predict
            preds = model.predict(X_input, verbose=0)
            
            # Calculate scores over time
            scores = np.mean(np.abs(preds - Y_actual), axis=(1, 2))
            
            # Get the MAX anomaly score (the "peak" fault level)
            max_score = np.max(scores)
            
            audit_results.append({
                'Unit': unit_id,
                'Max_Anomaly_Score': max_score,
                'Actual_RUL': df_rul[i]
            })

    # 3. Create DataFrame and Sort by Anomaly Score (Descending)
    audit_df = pd.DataFrame(audit_results)
    audit_df = audit_df.sort_values(by='Max_Anomaly_Score', ascending=False)

    return audit_df

def gen_train_windows_fd004(df, window_size, op_cols, sensor_cols):
    X, Y = [], []
    all_cols = op_cols + sensor_cols
    for unit in df['Unit'].unique():
        unit_data = df[df['Unit'] == unit]
        # Train only on the first 50 cycles (Healthy Baseline)
        input_data = unit_data[all_cols].values[:50]
        target_data = unit_data[sensor_cols].values[:50]
        
        if len(input_data) >= window_size:
            for i in range(len(input_data) - window_size + 1):
                X.append(input_data[i:i+window_size])
                Y.append(target_data[i:i+window_size])
    return np.array(X), np.array(Y)

def gen_test_windows_fd004(df_test, window_size, op_cols, sensor_cols):
    X, Y = [] , []
    # all_cols must match the 20 features used in training
    all_cols = op_cols + sensor_cols 
    
    for unit in df_test['Unit'].unique():
        unit_data = df_test[df_test['Unit'] == unit]
        
        # We take the LAST window available for each test engine
        # X needs the 3 Ops + 17 Sensors (20 total)
        # Y needs the 17 Sensors (to compare against model output)
        if len(unit_data) >= window_size:
            input_window = unit_data[all_cols].values[-window_size:]
            target_window = unit_data[sensor_cols].values[-window_size:]
            
            X.append(input_window)
            Y.append(target_window)
            
    return np.array(X), np.array(Y)

def anomaly_rul_plot(df, threshold):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(df['Actual_RUL'], df['Max_Anomaly_Score'], label='Model Results')

    def exponential_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    # 2. Generate dummy scatter data with noise
    x_data = np.linspace(np.min(df['Actual_RUL']), np.max(df['Actual_RUL']), 1000)

    # 3. Fit the curve
    # p0 is an optional initial guess for the parameters [a, b, c]
    initial_guess = [0.5, 0.1, 0.0]
    popt, pcov = curve_fit(exponential_decay, df['Actual_RUL'], df['Max_Anomaly_Score'], p0=initial_guess)

    # popt contains the optimized [a, b, c] values
    a_fit, b_fit, c_fit = popt
    print(a_fit, b_fit, c_fit)

    ax.plot(x_data, exponential_decay(x_data, *popt), orange, label='Exponential Model')
    
    ax.axhline(threshold, linestyle='--', label='Threshold', color=orange)

    ax.set_title('Model True RUL vs. Anomaly MAE')
    ax.set_xlabel('True RUL')
    ax.set_ylabel('Max Anomaly MAE')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', top=True, right=True, left=True, bottom=True, which='both', direction='in')
    ax.legend()

    fig.tight_layout()

    return fig

def plot_threshold_justification(train_mae_loss, threshold):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Plot the distribution
    sns.histplot(train_mae_loss, bins=50, kde=True, color='slategray')
    
    # 2. Add the Mean and Threshold lines
    mu = np.mean(train_mae_loss)
    ax.axvline(mu, color='blue', linestyle='--', label=f'Mean: {mu:.4f}')
    ax.axvline(threshold, color='red', linestyle='-', linewidth=2, label=f'Threshold (97.5\%): {threshold:.4f}')
    
    # 3. Shade the "Normal" zone
    ax.fill_betweenx([0, plt.gca().get_ylim()[1]], 0, threshold, color='green', alpha=0.1, label='Normal Operating Zone')

    ax.set_title(r'Training MAE Distribution \& Fault Threshold')
    ax.set_xlabel('Reconstruction Error (MAE)')
    ax.set_ylabel('Frequency (Number of Windows)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', top=True, right=True, left=True, bottom=True, which='both', direction='in')
    ax.set_xlim([np.min(train_mae_loss), np.max(train_mae_loss)])
    
    fig.tight_layout()
    return fig

def make_loss_plot(history):
    fig, ax = plt.subplots(figsize=(12, 6))

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    ax.plot(epochs, train_loss, blue, label='Training Loss')
    ax.plot(epochs, val_loss, orange, label='Validation Loss')
    
    ax.set_title('Training and Validation Loss (Autoencoder)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MAE)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    fig.tight_layout()

    return fig
def data_vis_hist(df, col: str, bins: int = 20, savefig: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(df[col], label=f'{col}', bins = bins)
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(f"{col}")
    ax.set_ylabel("Counts")
    ax.legend()

    fig.tight_layout()

    if savefig:
        fig.savefig(savefig)

    return fig

def data_vis_plot(df, col: str, bins: int = 20, savefig: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df['Cycle Time'], df[col], label=f'{col}')
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Sensor Reading")
    ax.legend()

    fig.tight_layout()

    if savefig:
        fig.savefig(savefig)

    return fig