"""
HackAI 2026 | Unsupervised LSTM Autoencoder for Jet Engine Fault Detection

Neil Ghugare, Nishanth Kunchala, and Jacob Balek
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from typing import *
from scipy.optimize import curve_fit

# Use colorblind palette and use TeX formatting for better-looking plots
plt.style.use('tableau-colorblind10')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16
})

# Fetch the blue and orange (first two colors from the palette) for better later use control
blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
orange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]


def failure_plot(anomaly_score: Any, threshold: Any, unit_no: int, savefig: str = None) -> plt.Figure:
    """
    Makes a plot of the anomaly MAE score as a function of cycle time, identifying the fault point, if it exists.
    
    Arguments (required)
    1. anomaly_score: The list of MAE values per cycle time.
    2. threshold: The fault detection threshold value.
    3. unit_no: The unit number of the engine.

    Arguments (optional)
    1. savefig: The string filename to save the figure to, with path. Defaults to None (no saving).
    
    Returns
    A matplotlib figure instance.
    """

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

def sensor_explore_plot(df: pd.DataFrame, ids: list, values: list) -> sns.FacetGrid:
    """
    Makes a Seaborn exploratory plot of the sensor and operational setting data as a function of
    cycle time for the first unit.
    
    Arguments (required)
    1. df: The DataFrame to use.
    2. ids: The x-axis (we are only going to use cycle time).
    3. values: The sensor variables to plot for.

    Arguments (optional)
    None
    
    Returns
    A matplotlib-compatible Seaborn FacetGrid instance.
    """
    unit_data = df[df['Unit'] == 1]
    plot_data = unit_data.melt(id_vars=ids, value_vars=values)

    # 2. Create a grid of plots
    g = sns.FacetGrid(plot_data, col="variable", col_wrap=4, sharey=False, aspect=1.5)
    g.map(sns.lineplot, "Cycle Time", "value")

    # Add a title and adjust layout
    g.figure.suptitle('Sensor Trends over Engine Life (Unit 1)', y=1.02, fontsize=16)

    return g

def gen_train_windows_multi(df, window_size, op_cols, sensor_cols):
    """
    Makes a plot of the anomaly MAE score as a function of cycle time, identifying the fault point, if it exists.
    
    Arguments (required)
    1. anomaly_score: The list of MAE values per cycle time.
    2. threshold: The fault detection threshold value.
    3. unit_no: The unit number of the engine.

    Arguments (optional)
    1. savefig: The string filename to save the figure to, with path. Defaults to None (no saving).
    
    Returns
    A matplotlib figure instance.
    """
    
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

def plot_threshold_justification(train_mae_loss, threshold, bins: int = 50):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Plot the distribution
    sns.histplot(train_mae_loss, bins=bins, kde=True, color='slategray')
    
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
    ax.set_xlim([np.min(train_mae_loss), np.percentile(train_mae_loss, 99.5)])
    
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

def gen_sequences_and_labels(data, window, feature_cols):
    sequences, cycles, ruls = [], [], []
    for unit_id in data['Unit'].unique():
        unit_data = data[data['Unit'] == unit_id]
        
        # Using the perfectly flattened sensor_cols now
        feature_vals = unit_data[feature_cols].values
        cycle_vals = unit_data['Cycle Time'].values
        rul_vals = unit_data['RUL'].values if 'RUL' in unit_data.columns else np.zeros(len(unit_data))
        
        if len(unit_data) >= window:
            for i in range(len(unit_data) - window):
                sequences.append(feature_vals[i:i+window])
                cycles.append(cycle_vals[i+window])
                ruls.append(rul_vals[i+window])
                
    return np.array(sequences), np.array(cycles), np.array(ruls)


def failure_detection_fd004(df, model, unit, rul_threshold, wsize, feature_cols):
    unit_mask = df['Unit'] == unit
    X_unit, cycles_unit, ruls_unit = gen_sequences_and_labels(df[unit_mask], wsize, feature_cols)

    unit_preds = model.predict(X_unit)
    unit_mae = np.mean(np.abs(unit_preds - X_unit), axis=(1, 2))

    return unit_mae, cycles_unit

def failure_plot_fd004(unit_mae: Any, threshold: Any, unit_no: int, savefig: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(unit_mae, label='Reconstruction Error (Anomaly Score)')
    ax.axhline(y=threshold, linestyle='--', color=orange, label='Quiet Failure Threshold')
    try:
        alert_idx = np.where(unit_mae > threshold)[0][0]
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

def anomaly_rul_plot_fd004(ruls, maes, threshold):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(ruls, maes, label='Model Results')

    def exponential_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    # 2. Generate dummy scatter data with noise
    x_data = np.linspace(np.min(ruls), np.max(ruls), 1000)

    # 3. Fit the curve
    # p0 is an optional initial guess for the parameters [a, b, c]
    initial_guess = [0.5, 0.1, 0.0]
    popt, pcov = curve_fit(exponential_decay, ruls, maes, p0=initial_guess)

    # popt contains the optimized [a, b, c] values
    a_fit, b_fit, c_fit = popt
    #print(a_fit, b_fit, c_fit)

    ax.plot(x_data, exponential_decay(x_data, *popt), orange, label='Exponential Model')
    
    ax.axhline(threshold, linestyle='--', label='Threshold', color=orange)

    ax.set_title('Model True RUL vs. Anomaly MAE')
    ax.set_xlabel('True RUL')
    ax.set_ylabel('Max Anomaly MAE')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', top=True, right=True, left=True, bottom=True, which='both', direction='in')
    ax.legend()
    ax.set_ylim([0, 5])

    fig.tight_layout()

    return fig

def plot_attention_heatmap(inference_model, x_data, sample_idx=0):
    """
    Generates predictions and plots the attention heatmap for a specific sample.
    """
    # Run inference to get predictions and  weights
    results = inference_model.predict(x_data)
    all_attention_weights = results[1]
    
    # Extract the 2D attention matrix
    attention_matrix = all_attention_weights[sample_idx]
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Draw heatmap
    ax = sns.heatmap(
        attention_matrix, 
        cmap="viridis",
        cbar_kws={'label': 'Relative Attention'}
    )
    
    # Labels
    plt.title(f"Attention Weights Heatmap (Window {sample_idx})", fontsize=14, pad=15)
    plt.xlabel("Encoder Time Steps", fontsize=12, labelpad=10)
    plt.ylabel("Decoder Time Steps", fontsize=12, labelpad=10)
    
    # Invert the Y-axis (step 0 is at top left)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()