"""Hypothèses versionnées, gelées avant production et jamais ajustées sur le test."""

from dataclasses import asdict, dataclass
import hashlib
import json


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Contract:
    persistence_s: float = 5.0
    progress_vehicle_lengths: float = 1.0
    nominal_followup_s: float = 30.0
    low_speed_mps: float = 0.15
    low_speed_fraction: float = 0.8
    movement_vehicle_lengths: float = 1.2
    reference_margin_m: float = 0.15
    minimum_intersection_area_m2: float = 0.0001
    minimum_observation_s: float = 35.0
    sample_s: float = 0.2
    version: str = "physical-C1-C7-movement-resource-v3"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != "version" and (
                not isinstance(value, (int, float)) or not 0 < value < float("inf")
            ):
                raise ValueError(f"Invalid contract parameter: {name}")
        if self.nominal_followup_s < self.persistence_s or self.low_speed_fraction > 1:
            raise ValueError(
                "Follow-up must cover persistence; speed fraction must be <= 1."
            )

    def record(self):
        return asdict(self)

    @property
    def sha256(self):
        return digest(self.record())
