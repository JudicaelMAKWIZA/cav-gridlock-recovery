"""Resource arithmetic only; these tests never construct a simulator."""

from scripts.assess_dense_capacity import scale_cpu, approach_slots


def test_cpu_reference_is_recovered_at_30():
    result = scale_cpu(120, 30)
    assert result["lower_s"] == result["upper_s"] == 120


def test_horizon_scales_the_same_workload_envelope():
    a = scale_cpu(120, 100, 110)
    b = scale_cpu(120, 100, 220)
    assert b["lower_s"] == 2 * a["lower_s"]
    assert b["upper_s"] == 2 * a["upper_s"]
    assert a["upper_s"] > a["lower_s"] > 120


def test_approach_storage_keeps_actual_body_and_gap():
    assert approach_slots(60, 4, 3) == 108
    assert approach_slots(100, 4, 3) == 180


def test_short_approaches_cannot_store_a_vehicle():
    assert approach_slots(2, 4, 3) == 0
