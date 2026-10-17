from collections import deque

import numpy as np

from config.config_VAD import VADConfig


class AudioSegmenter:
    def __init__(self, cfg: VADConfig):
        self.start_frames = cfg.ms_to_frames(cfg.start_confirm_ms)
        self.end_frames = cfg.ms_to_frames(cfg.end_silence_ms)
        self.pre_roll_frames = cfg.ms_to_frames(cfg.pre_roll_ms)
        self.tail_frames = cfg.ms_to_frames(cfg.tail_ms)
        self.max_frames = cfg.ms_to_frames(cfg.max_segment_ms)

        # deve contenere anche i frame usati per confermare l'inizio
        self.pre_roll = deque(maxlen=self.pre_roll_frames + self.start_frames)
        self.speech_frames = []
        self.pending_silence = []

        self.active_count = 0
        self.silence_count = 0
        self.total_frames = 0
        self.speaking = False

    def process(self, raw_frame, active):
        """
        Riceve un frame originale e il risultato del VAD.
        Restituisce (speech_started, completed_audio).
        completed_audio è None se la registrazione non è terminata.
        """
        speech_started = False
        completed_audio = None

        if not self.speaking:
            self.pre_roll.append(raw_frame.copy())

            if active:
                self.active_count += 1
            else:
                self.active_count = 0

            if self.active_count >= self.start_frames:
                self.speaking = True
                self.speech_frames = list(self.pre_roll)
                self.pending_silence = []
                self.silence_count = 0
                self.total_frames = len(self.speech_frames)
                speech_started = True

            return speech_started, completed_audio

        self.total_frames += 1

        if active:
            self.speech_frames.extend(self.pending_silence)
            self.pending_silence.clear()
            self.speech_frames.append(raw_frame.copy())
            self.silence_count = 0
        else:
            self.pending_silence.append(raw_frame.copy())
            self.silence_count += 1

        end_by_silence = self.silence_count >= self.end_frames
        end_by_length = self.total_frames >= self.max_frames

        if end_by_silence or end_by_length:
            tail = self.pending_silence[: self.tail_frames]
            completed_audio = np.concatenate(
                self.speech_frames + tail
            ).astype(np.float32)
            self._reset()

        return speech_started, completed_audio

    def _reset(self):
        self.pre_roll.clear()
        self.speech_frames = []
        self.pending_silence = []
        self.active_count = 0
        self.silence_count = 0
        self.total_frames = 0
        self.speaking = False

    def reset(self):
        self._reset()