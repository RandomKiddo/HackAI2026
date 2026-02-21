import matplotlib.pyplot as plt

from typing import *

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
    ax.set_title(f'Engine Unit {unit_no}: Quiet Failure Detection Over Time')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('MAE Error')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', top=True, right=True, left=True, bottom=True, which='both', direction='in')
    
    ax.legend()
    fig.tight_layout()

    if savefig:
        fig.savefig(savefig)

    return fig

