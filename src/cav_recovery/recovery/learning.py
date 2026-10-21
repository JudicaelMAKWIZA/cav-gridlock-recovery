"""Frontière minimale d'observation et masque d'actions, sans politique apprise."""

from .contracts import Action
from .protocol import validate
from .baseline import gentle_forward_options
import copy


def observation(view):
    """Ne projette ni l'audit, ni une graine, ni les membres physiques cachés."""
    for message in view.messages:
        validate(message, view.at_s)
    return copy.deepcopy(
        dict(
            own=[
                view.own[k] for k in ["speed_mps", "progress_m", "length_m", "width_m"]
            ],
            own_pose=dict(
                position=view.own["position"], heading_rad=view.own["heading_rad"]
            ),
            own_route_xy_m=list(view.route.line.coords)
            if view.route is not None
            else None,
            own_controls=view.own_controls,
            reverse_used_m=view.reverse_used_m,
            own_commitment_until_s=view.own_commitment_until_s,
            available_at_s=view.at_s,
            lidar=list(view.scan["ranges"]),
            lidar_available=not view.scan.get("missing", False),
            messages=[dict(m) for m in view.messages],
            diagnostic=getattr(view, "diagnostic", None),
            graph=getattr(view, "graph", None),
            measurement_contract=view.measurement_contract,
            scope="causal_local_only_no_reference_reward_or_global_membership",
        )
    )


def action_mask(view, guard, *, steering=0.0, attempt=None):
    options = [
        Action(v, steering, reason="learning_interface", attempt=attempt)
        for v in [0.0, 0.4, 0.8, -0.3, -0.6]
    ] + gentle_forward_options()
    return options, [guard.check(view, a)["accepted"] for a in options]
