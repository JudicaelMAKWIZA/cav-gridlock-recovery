"""Le TTL inclusif ne dépend pas de l'arrondi binaire de 11,6 + 1,2."""

from cav_recovery.metadrive.cooperation import MetaDriveExchange


def test_decimal_expiry_boundary_remains_valid_until_the_next_sample():
    exchange = MetaDriveExchange(
        range_m=30.0, interval_s=0.4, lifetime_s=1.2, step_s=0.2
    )
    for tick in range(1, 66):
        at_s = round(tick * 0.2, 7)
        rows = {
            key: dict(
                position_xy_m=[position, 0.0],
                heading_rad=0.0,
                signed_speed_mps=0.0,
                length_m=4.515,
                width_m=1.852,
                mission=dict(destination="east"),
            )
            for key, position in (("V0", 0.0), ("V1", 5.0 if at_s <= 11.6 else 100.0))
        }
        exchange.observe(at_s, rows)
        if at_s == 12.8:
            message = exchange.knowledge("V0")["received"]["V1"]["message"]
            assert message["observed_at_s"] == 11.6
            assert abs(message["valid_until_s"] - 12.8) < 1e-7
            assert "V1" not in exchange.knowledge("V0")["expired"]
    assert "V1" not in exchange.knowledge("V0")["received"]
    assert "V1" in exchange.knowledge("V0")["expired"]
