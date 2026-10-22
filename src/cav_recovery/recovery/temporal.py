"""Adaptation causale vers le modèle M02 gelé, sans réentraînement ni oracle."""

from ..laboratory.features import received_ages


def temporal_row(packet, diagnostic, graph):
    """Uniquement le paquet propre disponible et le graphe local de cet ego."""
    at = packet["decision_at_s"]
    controls = packet["own_controls"]
    return dict(
        ego_id=packet["ego_id"],
        movement_id=packet["mission"]["movement_id"],
        observed_at_s=at,
        available_at_s=at,
        own_state=packet["ego"],
        lidar=packet["lidar"]["normalized_ranges"],
        requested_speed_mps=controls["requested_speed_mps"],
        accepted_speed_mps=controls["accepted_speed_mps"],
        own_native_action=controls["native_action"],
        received_message_ages_s=received_ages(packet["v2v"], at),
        v2v_box_available=packet["v2v"] is not None,
        lidar_available=not packet["lidar"]["missing"],
        local_status=diagnostic["status"],
        current_received_relations=len(graph["edges"]),
        observed_collective_groups=len(graph["groups"]),
    )
