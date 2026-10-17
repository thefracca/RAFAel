"""
Layer 1: (mic, riferimento casse) -> registrazioni segmentate.

Ingresso: una sorgente con get_frame(timeout) -> (mic, ref), sample_rate, frame_size
          (di norma AudioDuplex, creato dal main)
Uscita:   np.ndarray, float32, mono, sample_rate di VADConfig, già senza eco

Il layer NON possiede lo stream: start/stop gestiscono solo il suo thread.
"""
import queue
import threading
from queue import Empty

from config.config_VAD import VADConfig
from .layer1_func.VAD_denoiser import AudioPreprocessor
from .layer1_func.vad_detector import VADDetector
from .layer1_func.audio_segmenter import AudioSegmenter
from .layer1_func.echo_canceller import EchoCanceller


class Layer1:
    def __init__(self, source, cfg: VADConfig = None):
        self.cfg = cfg or VADConfig()

        if source.sample_rate != self.cfg.sample_rate or source.frame_size != self.cfg.frame_size:
            raise ValueError(
                "AudioConfig e VADConfig non coincidono "
                f"(sample_rate {source.sample_rate}/{self.cfg.sample_rate}, "
                f"frame_size {source.frame_size}/{self.cfg.frame_size})"
            )
        self.source = source

        self.echo_canceller = EchoCanceller(self.cfg)
        self.preprocessor = AudioPreprocessor(self.cfg)
        self.detector = VADDetector(self.cfg)
        self.segmenter = AudioSegmenter(self.cfg)

        self.running = threading.Event()
        self.thread = None
        self.recordings = queue.Queue()

        # callback leggera: es. interrompere la riproduzione (barge-in)
        self.on_speech_start = lambda: None

    def _process_frame(self, mic, ref):
        clean = self.echo_canceller.process(mic, ref)

        # mentre l'assistente parla il VAD è meno sensibile (eco residuo)
        scale = self.cfg.playback_threshold_mult if self.echo_canceller.ref_active else 1.0

        filtered = self.preprocessor.process(clean)
        active = self.detector.is_active(filtered, scale)

        speech_started, completed_audio = self.segmenter.process(clean, active)

        if speech_started:
            self.on_speech_start()
        if completed_audio is not None:
            self.recordings.put(completed_audio)

    def _listen(self):
        while self.running.is_set():
            try:
                mic, ref = self.source.get_frame(timeout=0.1)
            except Empty:
                continue
            self._process_frame(mic, ref)

    def start(self):
        if self.running.is_set():
            return self
        self.echo_canceller.reset()
        self.preprocessor.reset()
        self.segmenter.reset()

        self.running.set()
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.running.clear()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def get_recording(self, timeout=None):
        """np.ndarray float32, oppure None se scade il timeout."""
        try:
            return self.recordings.get(timeout=timeout)
        except Empty:
            return None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()