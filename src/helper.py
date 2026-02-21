import matplotlib.pyplot as plt
import pandas as pd

from typing import *

plt.style.use('tableau-colorblind10')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16
})


def failure_plot(anomaly_score: Any, threshold: Any, savefig: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(anomaly_score, label='Reconstruction Error (Anomaly Score)')
    ax.axhline(y=threshold, linestyle='--', label='Quiet Failure Threshold')
    ax.set_title('Engine Unit 16: Quiet Failure Detection Over Time')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('MAE Error')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', top=True, right=True, left=True, bottom=True, which='both', direction='in')
    
    fig.tight_layout()

    if savefig:
        fig.savefig(savefig)

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
