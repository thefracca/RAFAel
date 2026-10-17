import numpy as np

from config.config_audio import AudioConfig
from config.config_VAD import VADConfig
from src.audio_io.audio_duplex import AudioDuplex
from src.layer1.layer1_func.echo_canceller import EchoCanceller

DURATA = 12
audio_cfg, vad_cfg = AudioConfig(), VADConfig()
sr = audio_cfg.sample_rate

rng = np.random.default_rng(0)
t = np.arange(DURATA * sr) / sr
sig = np.convolve(rng.standard_normal(len(t)), np.ones(4) / 4, mode="same")
sig *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
sig = (0.2 * sig / np.max(np.abs(sig))).astype(np.float32)

aec = EchoCanceller(vad_cfg)
mic_all, ref_all, clean_all = [], [], []

with AudioDuplex(audio_cfg) as duplex:
    duplex.play(sig)
    for _ in range(int((DURATA + 1) * 1000 / audio_cfg.frame_ms)):
        mic, ref = duplex.get_frame(timeout=1.0)
        clean = aec.process(mic, ref)
        mic_all.append(mic); ref_all.append(ref); clean_all.append(clean)

mic_v = np.concatenate(mic_all); ref_v = np.concatenate(ref_all)
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
half = len(mic_all) // 2
cl = np.concatenate(clean_all)

# solo dove il riferimento è attivo
mask = np.abs(ref_v) > 0
print(f"RMS riferimento: {rms(ref_v[mask]):.4f}")
print(f"RMS mic (con riferimento attivo): {rms(mic_v[mask]):.4f}")
print(f"rapporto mic/ref: {rms(mic_v[mask]) / (rms(ref_v[mask]) + 1e-9):.3f}  (soglia doppio parlato: {vad_cfg.aec_dt_ratio})")

# ritardo stimato (cross-correlazione su 4 s)
seg = slice(2 * sr, 6 * sr)
a, b = mic_v[seg], ref_v[seg]
corr = np.fft.irfft(np.fft.rfft(a, 2 * len(a)) * np.conj(np.fft.rfft(b, 2 * len(b))))
lag = int(np.argmax(corr[: sr // 2]))
peak = float(corr[lag] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print(f"ritardo eco: {1000 * lag / sr:.0f} ms, correlazione: {peak:.2f}  (filtro: {vad_cfg.aec_filter_ms} ms)")

print(f"frame adattati: {aec.adapt_frames}/{aec.total_frames}")
print(f"reset per divergenza: {aec.divergence_resets}")

mp = np.sum(np.square(mic_v[half * vad_cfg.frame_size:]))
cp = np.sum(np.square(cl[half * vad_cfg.frame_size:]))
print(f"ERLE: {10 * np.log10(mp / (cp + 1e-12)):.1f} dB")