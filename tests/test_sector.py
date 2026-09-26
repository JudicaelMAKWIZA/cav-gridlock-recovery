import json
from pathlib import Path
import tempfile
import unittest

from cav_recovery.empirical.sector import (
    AssociationParameters,
    SectorConfigurationError,
    TrackObservation,
    associate_segment,
    detect_crossings,
    load_sector,
)


FIXTURES = Path(__file__).parent / "fixtures" / "sector_profile"
DEGREE_PER_METER = 1.0 / 111_195.0


def observation(index, x_m, y_m, time_s, track_id="t"):
    return TrackObservation(2, index, track_id, "Car", y_m * DEGREE_PER_METER, x_m * DEGREE_PER_METER, time_s)


def parameters(**changes):
    values = dict(max_distance_m=6.0, max_branch_extent_m=60.0, max_heading_difference_deg=45.0, ambiguity_margin_m=1.0,
                  minimum_displacement_m=0.05, gate_hysteresis_m=0.5,
                  gate_rearm_distance_m=5.0, max_time_gap_s=2.0, max_space_gap_m=15.0)
    values.update(changes)
    return AssociationParameters(**values)


class SectorTests(unittest.TestCase):
    def setUp(self):
        self.sector = load_sector(FIXTURES / "sector_seed.json")

    def test_simple_crossing_and_dense_sampling_count_once(self):
        points = [observation(index, 0, y, index * 0.2) for index, y in enumerate([30, 25, 21, 20.2, 19.8, 15, 5, -5, -15, -19.8, -20.2, -25, -30])]
        crossings, ruptures = detect_crossings(points, self.sector, parameters())
        self.assertEqual([(event.gate_id, event.role) for event in crossings], [("N", "entry"), ("S", "exit")])
        self.assertEqual(ruptures, 0)
        self.assertTrue(all(event.interval_start_s <= event.estimated_time_s <= event.interval_end_s for event in crossings))

    def test_oscillation_near_gate_does_not_recount(self):
        points = [observation(index, 0, y, index * 0.2) for index, y in enumerate([25, 21, 19, 21, 19, 15, 10])]
        crossings, _ = detect_crossings(points, self.sector, parameters())
        self.assertEqual([event.gate_id for event in crossings], ["N"])

    def test_rupture_prevents_interpolated_crossing(self):
        points = [observation(0, 0, 25, 0), observation(1, 0, 15, 10)]
        crossings, ruptures = detect_crossings(points, self.sector, parameters())
        self.assertEqual(crossings, [])
        self.assertEqual(ruptures, 1)

    def test_spatial_jump_prevents_interpolated_crossing(self):
        points = [observation(0, 0, 25, 0), observation(1, 0, 5, 1)]
        crossings, ruptures = detect_crossings(points, self.sector, parameters(max_space_gap_m=10.0))
        self.assertEqual(crossings, [])
        self.assertEqual(ruptures, 1)

    def test_invalid_observation_barrier_prevents_crossing(self):
        points = [observation(0, 0, 25, 0), TrackObservation(2, 2, "t", "Car", 15 * DEGREE_PER_METER, 0, 1, True)]
        crossings, ruptures = detect_crossings(points, self.sector, parameters())
        self.assertEqual(crossings, [])
        self.assertEqual(ruptures, 1)

    def test_hysteresis_band_cannot_extend_interpolation_beyond_limits(self):
        points = [
            observation(0, 0, 21, 0.0),
            observation(1, 0, 20.2, 0.8),
            observation(2, 0, 20.1, 1.6),
            observation(3, 0, 19.9, 2.4),
            observation(4, 0, 19.0, 3.2),
        ]
        crossings, ruptures = detect_crossings(points, self.sector, parameters(max_time_gap_s=1.0))
        self.assertEqual(crossings, [])
        self.assertEqual(ruptures, 0)

    def test_branch_association_can_remain_ambiguous(self):
        previous = observation(0, -12, 12, 0)
        current = observation(1, -8, 8, 1)
        result = associate_segment(previous, current, self.sector, parameters(max_distance_m=15.0, max_heading_difference_deg=50.0))
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.branch_id)

    def test_association_is_bounded_to_the_useful_sector(self):
        previous = observation(0, -102, 102, 0)
        current = observation(1, -98, 98, 1)
        result = associate_segment(previous, current, self.sector, parameters(max_distance_m=150.0, max_heading_difference_deg=50.0, max_branch_extent_m=60.0))
        self.assertEqual(result.status, "outside")

    def test_distant_ambiguity_does_not_affect_local_association(self):
        configured = parameters(max_distance_m=150.0, max_heading_difference_deg=50.0, max_branch_extent_m=60.0)
        distant = associate_segment(observation(0, -102, 102, 0), observation(1, -98, 98, 1), self.sector, configured)
        local = associate_segment(observation(2, 0, 30, 2), observation(3, 0, 25, 3), self.sector, configured)
        self.assertEqual(distant.status, "outside")
        self.assertEqual(local.status, "accepted")

    def test_degenerate_gate_is_rejected(self):
        document = json.loads((FIXTURES / "sector_seed.json").read_text(encoding="utf-8"))
        document["gates"][0]["endpoints"][1] = document["gates"][0]["endpoints"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(SectorConfigurationError):
                load_sector(path)

    def test_parameters_reject_incoherent_hysteresis(self):
        with self.assertRaises(SectorConfigurationError):
            parameters(gate_hysteresis_m=5.0, gate_rearm_distance_m=1.0).validate()
