"""Le diagnostic reste passif et la carte transmise garde sa géométrie exacte."""

import json

import pytest

from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from cav_recovery.dataset.contract import Contract
from cav_recovery.dataset.native import simulate, local_drivable_regions
from cav_recovery.dataset.pipeline import read_rows
from cav_recovery.experiments.catalog import catalog, specification
from cav_recovery.experiments.session import DetectionSession


def test_local_map_decomposition_preserves_native_islands():
    surface = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(3, 3), (7, 3), (7, 7), (3, 7)]]
    )
    regions = local_drivable_regions(surface)
    assert all(not region.interiors for region in regions)
    combined = unary_union(regions)
    assert combined.equals(surface)
    assert not combined.covers(Point(5, 5))


def test_native_diagnostic_preserves_physics_controls_and_valid_static_map(tmp_path):
    spec = specification("collective-1", horizon_s=1.2)
    off = simulate(spec, tmp_path / "off", Contract())
    session = DetectionSession(tmp_path / "on")
    on = simulate(spec, tmp_path / "on", Contract(), observer=session)
    assert not off["integrity"]["errors"] and not on["integrity"]["errors"]
    for filename in ("physics.jsonl.gz", "commands.jsonl.gz"):
        assert list(read_rows(tmp_path / "off" / filename)) == list(
            read_rows(tmp_path / "on" / filename)
        )
    # Les UUID internes sont recréés par MetaDrive ; l'identité CAV stable reste agentN.
    # L'intégrité de chaque UUID pendant son propre épisode est contrôlée par simulate.
    events = []
    for mode in ("off", "on"):
        events.append(
            [
                {key: value for key, value in row.items() if key != "native_id"}
                for row in read_rows(tmp_path / mode / "native_events.jsonl.gz")
            ]
        )
    assert events[0] == events[1]
    assert off["integrity"]["introduced"] == on["integrity"]["introduced"]
    context = json.loads((tmp_path / "on/diagnostic/contexts.json").read_text("utf-8"))
    assert all(
        Polygon(points).is_valid
        for local in context.values()
        for points in local["map"]["drivable_polygons_xy_m"]
    )
    summary = json.loads((tmp_path / "on/diagnostic/summary.json").read_text("utf-8"))
    assert "INVALID_STATIC_GEOMETRY" not in summary["reasons"]
    assert (tmp_path / "on/diagnostic/features-contract.json").exists()


@pytest.mark.parametrize("name", [row["name"] for row in catalog()])
def test_four_native_topologies_have_valid_local_maps_and_render_offscreen(
    tmp_path, name
):
    from cav_recovery.experiments.view import ExperimentView

    class InspectionSession(DetectionSession):
        def observe(self, at_s, measurements, **kwargs):
            super().observe(at_s, measurements, **kwargs)
            if self.view is None:
                self.view = ExperimentView(window=False)
            self.view.inspect(self.last)
            import pygame

            pygame.image.save(self.view.screen_canvas, str(tmp_path / "inspection.png"))

    session = InspectionSession(tmp_path / name)
    result = simulate(
        specification(name, horizon_s=0.4),
        tmp_path / name,
        Contract(),
        observer=session,
    )
    assert not result["integrity"]["errors"]
    assert (tmp_path / "inspection.png").stat().st_size > 10000
    assert (
        session.statuses["UNKNOWN"] == 0
        or "INVALID_STATIC_GEOMETRY" not in session.reasons
    )
    contexts = json.loads(
        (tmp_path / name / "diagnostic/contexts.json").read_text("utf-8")
    )
    assert all(
        Polygon(points).is_valid
        for local in contexts.values()
        for points in local["map"]["drivable_polygons_xy_m"]
    )
