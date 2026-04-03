import matplotlib.pyplot import plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class TriggerVisualizer:
    """
    Provides publication-quality visualizations for the Trigger System.
    """
    def __init__(self):
        # Set overall seaborn style for publication quality
        sns.set_theme(style="whitegrid", context="paper")
        plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})
        
    def plot_energy_histogram(self, df: pd.DataFrame, triggered: np.ndarray, title: str = "Energy Distribution: Before vs After Filtering") -> plt.Figure:
        """
        Plots histograms showing the total energy distribution and the retained energy distribution.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # All events
        sns.histplot(df['energy'], bins=50, color='gray', alpha=0.5, label='All Events (Before Trigger)', ax=ax)
        
        # Accepted events
        sns.histplot(df['energy'][triggered == 1], bins=50, color='blue', alpha=0.7, label='Accepted Events (After Trigger)', ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Energy", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        return fig

    def plot_feature_scatter(self, df: pd.DataFrame, title: str = "Signal vs Noise: Features") -> plt.Figure:
        """
        Scatter plot to show separation between signal and background based on energy and noise level.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Use pandas mapping for legend
        df_plot = df.copy()
        df_plot['Type'] = df_plot['signal_label'].map({1: 'Signal (Real)', 0: 'Background (Noise)'})
        
        sns.scatterplot(
            data=df_plot, 
            x='noise_level', 
            y='energy', 
            hue='Type', 
            palette={'Signal (Real)': 'red', 'Background (Noise)': 'gray'},
            alpha=0.6,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Noise Level", fontsize=12)
        ax.set_ylabel("Energy", fontsize=12)
        
        plt.tight_layout()
        return fig
        
    def plot_trigger_decision(self, df: pd.DataFrame, triggered: np.ndarray, title: str = "Trigger Decisions") -> plt.Figure:
        """
        Scatter plot showing which events were retained and which were rejected.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        df_plot = df.copy()
        df_plot['Decision'] = np.where(triggered == 1, 'Accepted', 'Rejected')
        
        sns.scatterplot(
            data=df_plot, 
            x='noise_level', 
            y='energy', 
            hue='Decision', 
            palette={'Accepted': 'blue', 'Rejected': 'orange'},
            alpha=0.6,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Noise Level", fontsize=12)
        ax.set_ylabel("Energy", fontsize=12)
        
        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> plt.Figure:
        """
        Heatmap for confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Rejected', 'Accepted'], 
                    yticklabels=['Background', 'Signal'],
                    ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Trigger Prediction', fontsize=12)
        ax.set_ylabel('Ground Truth', fontsize=12)
        
        plt.tight_layout()
        return fig
