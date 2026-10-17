import numpy as np

from config.config_audio import AudioConfig
from config.config_VAD import VADConfig
from src.audio_io.audio_duplex import AudioDuplex
from src.layer1.layer1_func.echo_canceller import EchoCanceller

DURATA = 12   # secondi
audio_cfg, vad_cfg = AudioConfig(), VADConfig()
sr = audio_cfg.sample_rate

# segnale "tipo voce": rumore colorato con inviluppo lento
rng = np.random.default_rng(0)
t = np.arange(DURATA * sr) / sr
sig = np.convolve(rng.standard_normal(len(t)), np.ones(4) / 4, mode="same")
sig *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t) ** 2
sig = (0.2 * sig / np.max(np.abs(sig))).astype(np.float32)

aec = EchoCanceller(vad_cfg)
mic_all, clean_all = [], []

print("Stai zitto per", DURATA, "secondi...")
with AudioDuplex(audio_cfg) as duplex:
    duplex.play(sig)
    for _ in range(int((DURATA + 1) * 1000 / audio_cfg.frame_ms)):
        mic, ref = duplex.get_frame(timeout=1.0)
        clean = aec.process(mic, ref)
        mic_all.append(mic)
        clean_all.append(clean)

half = len(mic_all) // 2   # ignoro la fase di convergenza
mic_p = np.sum(np.square(np.concatenate(mic_all[half:])))
cln_p = np.sum(np.square(np.concatenate(clean_all[half:])))
print(f"ERLE (seconda metà): {10 * np.log10(mic_p / (cln_p + 1e-12)):.1f} dB")
print("reset per divergenza:", aec.divergence_resets)