from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    capture_queue_size: int = 100   # ~3 secondi
    input_device: object = None     # None = default di sistema
    output_device: object = None
    latency: str = "low"

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)