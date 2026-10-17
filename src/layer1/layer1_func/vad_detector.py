import numpy as np

from config.config_VAD import VADConfig


class VADDetector:
    def __init__(self, cfg: VADConfig):
        self.threshold = cfg.rms_threshold

    @staticmethod
    def calculate_rms(frame):
        frame = np.asarray(frame, dtype=np.float64)
        return float(np.sqrt(np.mean(frame ** 2)))

    def is_active(self, filtered_frame, threshold_scale=1.0):
        return self.calculate_rms(filtered_frame) >= self.threshold * threshold_scale