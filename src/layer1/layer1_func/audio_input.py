import queue

import sounddevice as sd

from config.config_VAD import (
    SAMPLE_RATE,
    CHANNELS,
    FRAME_SIZE,
    AUDIO_DTYPE,
    QUEUE_MAX_SIZE
)

class AudioInput:
    def __init__(self):
        self.audio_queue = queue.Queue(
            maxsize=QUEUE_MAX_SIZE
        )
        self.stream = None
        self.dropped_frames = 0

    def _callback(
            self,
            in_data,
            frame_count,
            time_info,
            status
    ):

        frame = indata[:, 0].copy()

        # scarto frame vecchio se si è in ritardo
        if self.audio_queue.full:
            try:
                self.audio_queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass

        try:
            self.audio_queue.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1

    def start(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
            blocksize=FRAME_SIZE,
            callback=self._callback,
        )
        self.stream.start()
        return self

    def get_frame(self):
        return self.audio_queue.get()

    def stop(self):
        return self.start()

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.stop()
