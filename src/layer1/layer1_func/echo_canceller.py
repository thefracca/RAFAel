from collections import deque

import numpy as np

from config.config_VAD import VADConfig


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


class EchoCanceller:
    """
    process(mic, ref) -> mic ripulito dall'eco (float32).
    ref_active è True mentre l'assistente parla (e per la coda di riverbero).
    Con aec_enabled=False non filtra ma continua a tracciare ref_active.
    """

    def __init__(self, cfg: VADConfig):
        self.enabled = cfg.aec_enabled
        self.B = cfg.frame_size
        self.N = 2 * self.B
        self.F = self.N // 2 + 1
        self.P = cfg.ms_to_frames(cfg.aec_filter_ms)
        self.mu = cfg.aec_step_size
        self.dt_ratio = cfg.aec_dt_ratio
        self.ref_min_rms = cfg.aec_ref_min_rms
        self.divergence_resets = 0
        self.adapt_frames = 0
        self.total_frames = 0
        self.reset()

    def reset(self):
        self.x_buf = np.zeros(self.N)
        self.X_hist = np.zeros((self.P, self.F), dtype=np.complex128)
        self.W = np.zeros((self.P, self.F), dtype=np.complex128)
        self.Px = np.full(self.F, 1e-3)
        self.ref_rms_hist = deque([0.0] * self.P, maxlen=self.P)
        self.ref_active = False

    def process(self, mic, ref):
        mic = np.asarray(mic, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        if len(mic) != self.B or len(ref) != self.B:
            raise ValueError(f"frame di {len(mic)}/{len(ref)} campioni, attesi {self.B}")

        self.ref_rms_hist.append(_rms(ref))
        ref_peak = max(self.ref_rms_hist)
        self.ref_active = ref_peak >= self.ref_min_rms

        if not self.enabled:
            self.total_frames += 1
            return mic.astype(np.float32)

        # aggiorna la storia del riferimento
        self.x_buf = np.concatenate((self.x_buf[self.B:], ref))
        self.X_hist = np.roll(self.X_hist, 1, axis=0)
        self.X_hist[0] = np.fft.rfft(self.x_buf)

        # eco stimato = convoluzione del riferimento con il filtro appreso
        Y = np.sum(self.W * self.X_hist, axis=0)
        y = np.fft.irfft(Y, n=self.N)[self.B:]
        e = mic - y

        # protezione anti-divergenza: l'uscita non deve mai essere molto
        # più forte dell'ingresso. Se succede, azzero il filtro.
        if self.ref_active and _rms(e) > 3.0 * _rms(mic) + 1e-4:
            self.W.fill(0.0)
            self.divergence_resets += 1
            return mic.astype(np.float32)

        # adatta solo se il riferimento c'è e non c'è doppio parlato (Geigel)
        double_talk = _rms(mic) > self.dt_ratio * ref_peak
        if self.ref_active and not double_talk:
            self.adapt_frames += 1
            self._adapt(e)

        return np.clip(e, -1.0, 1.0).astype(np.float32)

    def _adapt(self, e):
        E = np.fft.rfft(np.concatenate((np.zeros(self.B), e)))

        # potenza TOTALE su tutte le partizioni (non la media)
        power = np.sum(np.abs(self.X_hist) ** 2, axis=0)
        self.Px = 0.9 * self.Px + 0.1 * power

        # regolarizzazione relativa: evita gradienti enormi nei bin quasi vuoti
        delta = 0.01 * float(np.mean(power)) + 1e-3
        denom = np.maximum(self.Px, power) + delta

        grad = np.conj(self.X_hist) * (E / denom)[None, :]
        g = np.fft.irfft(grad, n=self.N, axis=1)
        g[:, self.B:] = 0.0                      # vincolo sul gradiente
        self.W += self.mu * np.fft.rfft(g, axis=1)