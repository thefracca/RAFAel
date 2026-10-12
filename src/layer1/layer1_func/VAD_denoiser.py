import numpy as np
from scipy.signal import butter, sosfilt

from config.config_VAD import (
    SAMPLE_RATE,
    HIGH_PASS_CUTOFF_HZ,
    FILTER_ORDER,
)

class AudioPreprocessor:
    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        cutoff_hz=HIGH_PASS_CUTOFF_HZ,
        order = FILTER_ORDER
    ):
        self.sos = butter(
            order,
            cutoff_hz,
            btype='highpass',
            fs=sample_rate,
            output='sos'
        )

        self.state = np.zeros(
            (self.sos.shape[0], 2),
            dtype=np.float32,
        )

    def process(self, frame):
        filtered, self.state = sosfilt(
            self.sos,
            frame,
            zi=self.state,
        )
        return filtered.astype(np.float32)

    def reset(self):
        self.state.fill(0.0)
