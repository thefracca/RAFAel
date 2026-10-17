from scipy.io import wavfile

from config.config_audio import AudioConfig
from config.config_VAD import VADConfig
from src.audio_io.audio_duplex import AudioDuplex
from src.layer1.main_layer1 import Layer1

vad_cfg = VADConfig(rms_threshold=0.01)   # cambia qui per provare
audio_cfg = AudioConfig()

with AudioDuplex(audio_cfg) as duplex, Layer1(duplex, vad_cfg) as layer1:
    layer1.on_speech_start = lambda: print("voce rilevata...")
    print("Parla, Ctrl+C per uscire")
    n = 0
    try:
        while True:
            audio = layer1.get_recording(timeout=0.1)
            if audio is None:
                continue
            n += 1
            wavfile.write(f"test_{n}.wav", vad_cfg.sample_rate, audio)
            print(f"salvato test_{n}.wav ({len(audio) / vad_cfg.sample_rate:.1f}s)")
    except KeyboardInterrupt:
        pass