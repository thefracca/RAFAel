# test_mic_hears.py
import numpy as np

from config.config_audio import AudioConfig
from src.audio_io.audio_duplex import AudioDuplex

cfg = AudioConfig()
sr = cfg.sample_rate
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))


def collect(duplex, seconds):
    n = int(seconds * 1000 / cfg.frame_ms)
    m, r = [], []
    for _ in range(n):
        mic, ref = duplex.get_frame(timeout=1.0)
        m.append(mic)
        r.append(ref)
    return np.concatenate(m), np.concatenate(r)


# sweep 300 -> 3400 Hz in 4 secondi
T = 4
t = np.arange(T * sr) / sr
phase = 2 * np.pi * (300 * t + (3400 - 300) * t ** 2 / (2 * T))
sweep = (0.3 * np.sin(phase)).astype(np.float32)

with AudioDuplex(cfg) as d:
    print("Silenzio per 2 secondi...")
    sil_mic, _ = collect(d, 2)
    print("Suono dalle casse...")
    d.play(sweep)
    mic, ref = collect(d, T + 0.5)

floor = rms(sil_mic)
active = np.abs(ref) > 0
level = rms(mic[active])
print(f"rumore di fondo mic: {floor:.5f}")
print(f"mic durante il suono: {level:.5f}")
print(f"differenza: {20 * np.log10(level / (floor + 1e-9)):.1f} dB")

# correlazione mic/riferimento, ritardo 0-300 ms
a, b = mic[sr:3 * sr], ref[sr:3 * sr]
c = np.fft.irfft(np.fft.rfft(a, 2 * len(a)) * np.conj(np.fft.rfft(b, 2 * len(b))))
lag = int(np.argmax(np.abs(c[: int(0.3 * sr)])))
corr = abs(c[lag]) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
print(f"ritardo: {1000 * lag / sr:.0f} ms, correlazione: {corr:.2f}")