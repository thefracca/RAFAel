from collections import deque
import numpy as np

from config.config_VAD import (
    START_CONFIRM_MS,
    END_SILENCE_MS,
    PRE_ROLL_MS,
    FRAME_MS,
)



class AudioSegmenter():
    def __init__(self):
        self.start_frames = max(
            1, round(START_CONFIRM_MS / FRAME_MS)
        )
        self.end_frames= max(
            1, round(END_SILENCE_MS / FRAME_MS)
        )
        self.pre_roll_frames = max(
            1, round(PRE_ROLL_MS / FRAME_MS)
        )

        self.pre_roll = deque(maxlen=self.pre_roll_frames)
        self.speech_frames = []
        self.pending_silence = []

        self.active_count = 0
        self.silence_count = 0
        self.speaking = False

    def process(self, raw_frame, active):
        """
        Riceve un frame originale e il risultato del VAD.

        Restituisce:
            (speech_started, completed_audio)

        completed_audio è None se la registrazione non è terminata.
        """

        speech_started = False
        completed_audio = None

        if not self.speaking:
            self.pre_roll.appendleft(raw_frame.copy())

            if active:
                self.active_count += 1
            else:
                self.active_count = 0

            if self.active_count >= self.start_frames:
                self.speaking = True
                self.speech_frames = list(self.pre_roll)
                self.prending_silence = []
                self.silence_count = 0
                speech_started = True

            return speech_started, completed_audio

        if active:
            self.speech_frames.extend(self.pending_silence)
            self.pending_silence.clear()

            self.speech_frames.append(raw_frame.copy())
            self.silence_count = 0

        else:
            self.pending_silence.append(raw_frame.copy())
            self.silence_count += 1

            if self.silence_count >= self.end_frames:
                completed_audio = np.concatenate(
                    self.speech_frames
                ).astype(np.float32)

                self._reset()

        return speech_started, completed_audio

    def _reset(self):
        self.pre_roll.clear()
        self.speech_frames = []
        self.pending_silence = []

        self.active_count = 0
        self.silence_count = 0
        self.speaking = False

    def reset(self):
        self._reset()
