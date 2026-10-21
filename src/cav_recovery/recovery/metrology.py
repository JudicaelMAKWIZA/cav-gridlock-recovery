"""Adaptation expérimentale de mesures, sans modifier l'observateur K04."""

import copy
import math


class StationaryMeasurement:
    """Borne un plateau immobile ; ne masque pas un recul commandé ou cumulatif.

    Les bornes sont celles de REC-METRO-1, distinctes du contrat historique.
    La pose physique n'est jamais changée. La mesure brute reste dans le reçu.
    """

    version = "REC-METRO-1"

    def __init__(self, speed_error_mps=0.01, position_error_m=0.01):
        if speed_error_mps != 0.01 or position_error_m != 0.01:
            raise ValueError(
                "REC-METRO-1 fixe ces deux bornes ; créer une autre version pour les changer."
            )
        self.speed_error = speed_error_mps
        self.position_error = position_error_m
        self.previous = None
        self.anchor = None
        self.corrected_progress = None
        self.rejected_hold = False

    def adapt(self, local):
        result = copy.deepcopy(local)
        own, controls = result["ego"], result["own_controls"]
        at = result["decision_at_s"]
        raw_progress = own["mission_progress_m"]
        speed = own["signed_speed_mps"]
        consecutive = (
            self.previous is not None and abs(at - self.previous[0] - 0.2) < 1e-7
        )
        held = (
            controls["accepted_speed_mps"] == 0 and controls["requested_speed_mps"] >= 0
        )
        held = held and abs(speed) <= self.speed_error
        if not consecutive:
            self.anchor = None
            self.corrected_progress = raw_progress
            self.rejected_hold = False
        if not held:
            self.rejected_hold = False
        if held and self.anchor is None and not self.rejected_hold:
            self.anchor = (list(own["position_xy_m"]), raw_progress)
        bounded = (
            held
            and self.anchor is not None
            and (
                math.dist(self.anchor[0], own["position_xy_m"]) <= self.position_error
                and abs(raw_progress - self.anchor[1]) <= self.position_error
            )
        )
        if bounded:
            own["signed_speed_mps"] = 0.0
            own["mission_progress_m"] = self.corrected_progress
        else:
            if held and self.anchor is not None:
                self.rejected_hold = True
            self.anchor = None
            if consecutive:
                self.corrected_progress += raw_progress - self.previous[1]
            else:
                self.corrected_progress = raw_progress
            own["mission_progress_m"] = self.corrected_progress
        self.previous = (at, raw_progress)
        receipt = dict(
            measurement_contract=self.version,
            bounded_stationary_interval=bounded,
            raw_speed_mps=speed,
            raw_progress_m=raw_progress,
            used_speed_mps=own["signed_speed_mps"],
            used_progress_m=own["mission_progress_m"],
            position_unchanged=True,
        )
        return result, receipt
