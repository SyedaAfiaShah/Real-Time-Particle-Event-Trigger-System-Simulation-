import pandas as pd
import numpy as np

class RuleBasedTrigger:
    """
    A simple threshold-based trigger system.
    """
    def __init__(self, energy_threshold: float = 30.0, max_noise_threshold: float = 15.0):
        """
        Initialize with basic physics thresholds.
        
        Args:
            energy_threshold: Minimum energy required to trigger.
            max_noise_threshold: Maximum allowable noise level to accept the event.
        """
        self.energy_threshold = energy_threshold
        self.max_noise_threshold = max_noise_threshold

    def set_thresholds(self, energy_threshold: float, max_noise_threshold: float):
        self.energy_threshold = energy_threshold
        self.max_noise_threshold = max_noise_threshold

    def evaluate(self, events: pd.DataFrame) -> np.ndarray:
        """
        Evaluate events against the rule-based criteria.
        
        Args:
            events (pd.DataFrame): Dataframe containing 'energy' and 'noise_level'.
            
        Returns:
            np.ndarray: Binary array where 1 means Trigger Accepted, 0 means Rejected.
        """
        accepted = (events['energy'] > self.energy_threshold) & \
                   (events['noise_level'] < self.max_noise_threshold)
        
        return accepted.astype(int).values
