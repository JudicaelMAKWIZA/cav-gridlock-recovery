import json
from pathlib import Path
import tempfile
import unittest

from cav_recovery.empirical.sector import (
    CrossingParameters,
    SectorConfigurationError,
    TrackObservation,
    detect_crossings,
    load_sector,
)


FIXTURES = Path(__file__).parent / "fixtures" / "sector_profile"
DEGREE_PER_METER = 1.0 / 111_195.0


def observation(index, x_m, y_m, time_s, track_id="t"):
    return TrackObservation(2, index, track_id, "Car", y_m * DEGREE_PER_METER, x_m * DEGREE_PER_METER, time_s)


def parameters(**changes):
    values = dict(deduplication_s=1.0, max_time_gap_s=2.0, max_space_gap_m=15.0)
    values.update(changes)
    return CrossingParameters(**values)


class SectorTests(unittest.TestCase):
    def setUp(self):
        self.sector = load_sector(FIXTURES / "sector_seed.json")

    def test_simple_crossing_and_dense_sampling_count_once(self):
        points = [observation(index, 0, y, index * 0.2) for index, y in enumerate([30, 25, 21, 20.2, 19.8, 15, 5, -5, -15, -19.8, -20.2, -25, -30])]
        crossings, ruptures = detect_crossings(points, self.sector, parameters())
        self.assertEqual([(event.gate_id, event.role) for event in crossings], [("N", "entry"), ("S", "exit")])
        self.assertEqual(ruptures, 0)
        self.assertTrue(all(event.interval_start_s <= event.estimated_time_s <= event.interval_end_s for event in crossings))

    def test_turn_crosses_one_entry_and_one_exit(self):
        coordinates = [(0, 30), (0, 21), (0, 19), (0, 5), (5, 0), (19, 0), (21, 0), (30, 0)]
        points = [observation(index, x_m, y_m, index * 0.2) for index, (x_m, y_m) in enumerate(coordinates)]
        crossings, _ = detect_crossings(points, self.sector, parameters())
        self.assertEqual([(event.gate_id, event.role) for event in crossings], [("N", "entry"), ("E", "exit")])

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

    def test_slow_progression_uses_only_the_pair_that_crosses(self):
        points = [
            observation(0, 0, 21, 0.0),
            observation(1, 0, 20.2, 0.8),
            observation(2, 0, 20.1, 1.6),
            observation(3, 0, 19.9, 2.4),
            observation(4, 0, 19.0, 3.2),
        ]
        crossings, ruptures = detect_crossings(points, self.sector, parameters(max_time_gap_s=1.0))
        self.assertEqual(len(crossings), 1)
        self.assertEqual((crossings[0].from_group_index, crossings[0].to_group_index), (2, 3))
        self.assertEqual(ruptures, 0)

    def test_reverse_crossing_is_not_counted(self):
        points = [observation(0, 0, 15, 0), observation(1, 0, 25, 1)]
        crossings, _ = detect_crossings(points, self.sector, parameters())
        self.assertEqual(crossings, [])

    def test_distant_motion_cannot_cross_a_finite_gate(self):
        points = [observation(0, -100, 25, 0), observation(1, -100, 15, 1)]
        crossings, _ = detect_crossings(points, self.sector, parameters())
        self.assertEqual(crossings, [])

    def test_near_gate_without_crossing_is_negative(self):
        points = [observation(0, -5, 21, 0), observation(1, 5, 21, 1)]
        crossings, _ = detect_crossings(points, self.sector, parameters())
        self.assertEqual(crossings, [])

    def test_two_passages_separated_by_deduplication_are_retained(self):
        points = [
            observation(0, 0, 21, 0),
            observation(1, 0, 19, 0.2),
            observation(2, 0, 21, 1.2),
            observation(3, 0, 19, 2.2),
        ]
        crossings, _ = detect_crossings(points, self.sector, parameters())
        self.assertEqual([event.gate_id for event in crossings], ["N", "N"])

    def test_non_increasing_time_breaks_continuity(self):
        points = [observation(0, 0, 25, 1), observation(1, 0, 15, 0.5)]
        crossings, ruptures = detect_crossings(points, self.sector, parameters())
        self.assertEqual(crossings, [])
        self.assertEqual(ruptures, 1)

    def test_detection_is_deterministic(self):
        points = [observation(index, 0, y, index * 0.2) for index, y in enumerate([25, 21, 19, 10])]
        first = detect_crossings(points, self.sector, parameters())
        second = detect_crossings(points, self.sector, parameters())
        self.assertEqual(first, second)

    def test_degenerate_gate_is_rejected(self):
        document = json.loads((FIXTURES / "sector_seed.json").read_text(encoding="utf-8"))
        document["gates"][0]["endpoints"][1] = document["gates"][0]["endpoints"][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(SectorConfigurationError):
                load_sector(path)

    def test_parameters_reject_non_positive_threshold(self):
        with self.assertRaises(SectorConfigurationError):
            parameters(deduplication_s=0.0).validate()
