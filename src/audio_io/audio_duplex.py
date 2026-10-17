"""
Stream full-duplex condiviso (mic + casse).

Ogni frame catturato viene accoppiato al frame che in QUELLO STESSO callback
va alle casse: è il segnale di riferimento per l'echo cancellation.

Contratto:
    get_frame() -> (mic, ref), entrambi np.float32 di frame_size campioni
    play(audio) -> audio np.float32 mono, già a sample_rate (il layer 5 deve ricampionare)
"""
import queue
import threading

import numpy as np
import sounddevice as sd

from config.config_audio import AudioConfig


class AudioDuplex:
    def __init__(self, cfg: AudioConfig = None):
        self.cfg = cfg or AudioConfig()
        self.sample_rate = self.cfg.sample_rate
        self.frame_size = self.cfg.frame_size

        self._capture = queue.Queue(maxsize=self.cfg.capture_queue_size)
        self._play_buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self.stream = None

        self.dropped_frames = 0
        self.stream_errors = 0

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.stream_errors += 1

        mic = indata[:, 0].copy()

        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            n = min(frames, len(self._play_buf))
            if n:
                out[:n] = self._play_buf[:n]
                self._play_buf = self._play_buf[n:]
        outdata[:, 0] = out

        if self._capture.full():
            try:
                self._capture.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
        try:
            self._capture.put_nowait((mic, out))
        except queue.Full:
            self.dropped_frames += 1

    # ---------- cattura ----------
    def get_frame(self, timeout=None):
        return self._capture.get(timeout=timeout)

    # ---------- riproduzione ----------
    def play(self, audio):
        audio = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
        with self._lock:
            self._play_buf = np.concatenate((self._play_buf, audio))

    def stop_playback(self):
        """Svuota subito il buffer (barge-in)."""
        with self._lock:
            self._play_buf = np.zeros(0, dtype=np.float32)

    @property
    def is_playing(self):
        with self._lock:
            return len(self._play_buf) > 0

    # ---------- ciclo di vita ----------
    def start(self):
        if self.stream is not None:
            return self
        self.stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            device=(self.cfg.input_device, self.cfg.output_device),
            channels=1,
            dtype="float32",
            latency=self.cfg.latency,
            callback=self._callback,
        )
        self.stream.start()
        return self

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()