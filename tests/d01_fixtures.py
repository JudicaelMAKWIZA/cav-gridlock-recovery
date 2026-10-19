"""Independent analytical F01 observations, never imported by production code."""

import copy


def packet(t=.4, *, ego_x=0., other_x=5., speed=0., other_speed=0., sender="V1"):
    t = round(t, 7)
    sent = round(.4 * int((t + 1e-7) / .4), 7)
    seq = max(1, round(sent / .4))
    return dict(schema_version="F01-input-1", contract_id="K04_APPROVED", ego_id="V0", decision_at_s=t,
        ego=dict(observed_at_s=t, available_at_s=t, position_xy_m=[ego_x, 0.], heading_rad=0.,
            signed_speed_mps=speed, velocity_xy_mps=[speed, 0.], acceleration_xy_mps2=[0., 0.],
            mission_progress_m=ego_x, length_m=4.515, width_m=1.852),
        mission=dict(observed_at_s=0., available_at_s=0., version="own-plan-1", origin="west", destination="east",
            own_route_xy_m=[[-20., 0.], [100., 0.]], movement_id="V0:own-plan-1:forward", demand_forward=True),
        map=dict(observed_at_s=0., available_at_s=0., version="fixture-cross-1", coordinate_frame="metadrive_local_xy_m",
            drivable_polygons_xy_m=[[[-20., -1.75], [100., -1.75], [100., 1.75], [-20., 1.75]],
                [[10.25, -20.], [13.75, -20.], [13.75, 20.], [10.25, 20.]]], lane_width_m=3.5, rule_context="not_modelled"),
        lidar=dict(observed_at_s=t, available_at_s=t, source="native_lidar", num_lasers=16, range_m=30.,
            height_m=1.2, phase_offset_rad=0., normalized_ranges=[round(min(1., (other_x-ego_x-2.2575)/30), 6)]+[1.]*15, missing=False),
        own_controls=dict(observed_at_s=t, available_at_s=t, requested_speed_mps=4., accepted_speed_mps=0., native_action=[0., 0.]),
        v2v=dict(available_at_s=t, received=[dict(received_at_s=sent, message=dict(protocol_version=1,
            message_id=f"{sender}:{seq}", sender_id=sender, sequence_number=seq, observed_at_s=sent, sent_at_s=sent,
            valid_until_s=round(sent+1.2, 7), state_schema_version="cgr-metadrive-state-1", coordinate_frame="metadrive_local_xy_m",
            position_xy_m=[other_x, 0.], heading_rad=0., signed_speed_mps=other_speed, length_m=4.515, width_m=1.852,
            intended_destination="east", progress_delta_m=0., progress_kind="unsigned_distance_not_mission_progress",
            progress_window_s=.2, source="metadrive_ego_emulation_not_radio", confidence="simulated_exact_ego"))], expired=[]))


def fixture_a():
    return [packet(round(.4+.2*n, 7)) for n in range(151)]


def fixture_b():
    result = []; previous = None
    for n in range(41):
        t = round(.4+.2*n, 7)
        x1 = 5+max(0., t-1.4)**2; x0 = .5*max(0., t-2.4)**2
        speed = max(0., t-2.4)
        p = packet(t, ego_x=x0, other_x=x1, speed=speed, other_speed=2*max(0., t-1.4))
        p["ego"]["acceleration_xy_mps2"] = [float(t>2.4), 0.]
        c = p["own_controls"]; c["requested_speed_mps"] = 4. if t<=2.4 else 6.
        c["accepted_speed_mps"] = speed; c["native_action"] = [0., .4 if t>2.4 else 0.]
        if n%2==0 and x1-x0<=30:
            m=p["v2v"]["received"][0]["message"]
            m["progress_delta_m"] = x1-(5+max(0., t-.2-1.4)**2)
            previous=copy.deepcopy(p["v2v"]["received"][0])
        p["v2v"]["received"]=[] if previous is None or t>previous["message"]["valid_until_s"]+1e-7 else [copy.deepcopy(previous)]
        result.append(p)
    return result


def fixture_c():
    result = fixture_a(); first=copy.deepcopy(result[0]["v2v"]["received"][0])
    for p in result[1:]:
        if p["decision_at_s"]<=1.6:
            p["v2v"]["received"]=[copy.deepcopy(first)]
        else:
            p["v2v"]["received"]=[]
            p["v2v"]["expired"]=[dict(sender_id="V1", message_id="V1:1", last_received_at_s=.4, valid_until_s=1.6)]
    return result
