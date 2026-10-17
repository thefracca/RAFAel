from dataclasses import dataclass


@dataclass(frozen=True)
class VADConfig:
    # Audio (devono coincidere con AudioConfig, Layer1 lo verifica)
    sample_rate: int = 16000
    frame_ms: int = 30

    # Pulizia (high-pass)
    high_pass_hz: int = 100
    filter_order: int = 4

    # VAD
    rms_threshold: float = 0.01

    # Segmentazione
    start_confirm_ms: int = 90
    end_silence_ms: int = 700
    pre_roll_ms: int = 300
    tail_ms: int = 150
    max_segment_ms: int = 30000

    # Echo cancellation
    aec_enabled: bool = True
    aec_filter_ms: int = 500         # deve coprire latenza hardware + riverbero
    aec_step_size: float = 0.3        # più alto = converge prima ma è più instabile
    aec_dt_ratio: float = 3.0        # sopra questo rapporto mic/ref = doppio parlato, non adatta
    aec_ref_min_rms: float = 0.002    # sotto questo il riferimento è considerato silenzio
    playback_threshold_mult: float = 3.0   # soglia VAD x3 mentre l'assistente parla (1.0 = off)

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    def ms_to_frames(self, ms: int) -> int:
        return max(1, round(ms / self.frame_ms))