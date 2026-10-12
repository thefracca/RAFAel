import numpy as np

from config.config_VAD import RMS_THRESHOLD


class VADDetector:
    def __init__(self, threshold=RMS_THRESHOLD):
        self.threshold = threshold

    @staticmethod
    def calculate_rms(frame):
        frame = np.asarray(frame, dtype=np.float64)
        return float(np.sqrt(np.mean(frame ** 2)))

    def is_active(self, filtered_frame):
        rms = self.calculate_rms(filtered_frame)
        return rms >= self.threshold