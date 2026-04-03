import pandas as pd
import numpy as np
from scipy import stats

class StatisticalTrigger:
    """
    A statistical trigger that models the background (noise) distribution 
    and accepts anomalies based on a p-value equivalent threshold.
    """
    def __init__(self, p_value_threshold: float = 0.05):
        """
        Args:
            p_value_threshold: The probability threshold under the background model
                               below which an event is flagged as 'signal'.
        """
        self.p_value_threshold = p_value_threshold
        self.bg_energy_mean = 0.0
        self.bg_energy_std = 1.0
        self.bg_momentum_mean = 0.0
        self.bg_momentum_std = 1.0
        self.fitted = False

    def fit(self, background_events: pd.DataFrame):
        """
        Fit the background model using pure background events (or mostly background).
        For simplicity, we fit Gaussian parameters, though realistically an 
        exponential or Gamma fits better. We use robust statistics.
        """
        self.bg_energy_mean = background_events['energy'].median()
        self.bg_energy_std = background_events['energy'].std()
        
        self.bg_momentum_mean = background_events['momentum'].median()
        self.bg_momentum_std = background_events['momentum'].std()
        
        # Prevent zero division
        self.bg_energy_std = max(self.bg_energy_std, 1e-4)
        self.bg_momentum_std = max(self.bg_momentum_std, 1e-4)

        self.fitted = True

    def set_thresholds(self, p_value_threshold: float):
        self.p_value_threshold = p_value_threshold

    def evaluate(self, events: pd.DataFrame) -> np.ndarray:
        """
        Evaluate events calculating the combined survival function (1 - CDF).
        Assumes high energy/momentum are anomalies (signals).
        """
        if not self.fitted:
            raise ValueError("Statistical trigger must be fitted prior to evaluation.")

        # Compute z-scores
        z_energy = (events['energy'] - self.bg_energy_mean) / self.bg_energy_std
        z_momentum = (events['momentum'] - self.bg_momentum_mean) / self.bg_momentum_std
        
        # p-values (right-tailed, we look for high energy and momentum)
        p_energy = stats.norm.sf(z_energy)
        p_momentum = stats.norm.sf(z_momentum)
        
        # We can combine p-values or simply use the product or min
        # Fisher's method could be used, here we just use the joint probability 
        # (assuming independence for the background model)
        joint_p_value = p_energy * p_momentum
        
        accepted = joint_p_value < self.p_value_threshold
        return accepted.astype(int).values
