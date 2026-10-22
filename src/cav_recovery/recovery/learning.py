"""Frontière minimale d'observation et masque d'actions, sans politique apprise."""

from .contracts import Action
from .protocol import validate
from .cooperative import option_bank
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


def action_mask(view, guard, *, steering=None, attempt=None, for_intention=False):
    """Le masque d'intention ne vaut pas autorisation d'exécution sans accord."""
    options = [
        Action(a.speed_mps, a.steering, a.reason, attempt)
        for a in option_bank()
        if steering is None or a.speed_mps == 0 or a.steering == steering
    ]
    return options, [
        guard.check(view, a, require_agreement=not for_intention)["accepted"]
        for a in options
    ]
