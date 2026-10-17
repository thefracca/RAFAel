import numpy as np
from scipy.signal import butter, sosfilt

from config.config_VAD import VADConfig


class AudioPreprocessor:
    def __init__(self, cfg: VADConfig):
        self.sos = butter(
            cfg.filter_order,
            cfg.high_pass_hz,
            btype="highpass",
            fs=cfg.sample_rate,
            output="sos",
        )
        self.state = np.zeros((self.sos.shape[0], 2), dtype=np.float64)

    def process(self, frame):
        filtered, self.state = sosfilt(self.sos, frame, zi=self.state)
        return filtered.astype(np.float32)

    def reset(self):
        self.state.fill(0.0)