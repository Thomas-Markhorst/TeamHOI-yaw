"""Table-heading task math. Pure Torch so it can be validated without Isaac Gym."""
import math
import torch

YAW_EXTRA_DIM = 4
LEGACY_GOAL_DIM = 3


def yaw_settings(config):
    defaults = dict(enabled=False, position_tolerance=0.03, yaw_tolerance_deg=5.0,
                    angular_speed_tolerance=0.1, angular_speed_scale=1.0,
                    braking_width_deg=15.0, accuracy_weight=1.0,
                    braking_weight=0.1, position_weight=0.25, position_width=0.1,
                    initial_yaw_range_deg=5.0, target_offset_range_deg=30.0,
                    settled_linear_speed=0.05, settled_steps=15, settled_height_tolerance=0.05,
                    reference_loss_coef=1.0, staged_transport=False,
                    carry_heading_weight=0.0, carry_angular_speed_weight=0.0,
                    carry_heading_scale_deg=30.0, lift_height=0.08,
                    in_place_probability=0.0, in_place_min_yaw_deg=10.0)
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError("Unknown yaw settings: {}".format(sorted(unknown)))
    defaults.update(config)
    for key in ("position_tolerance", "yaw_tolerance_deg", "angular_speed_scale",
                "braking_width_deg", "position_width", "settled_steps", "settled_height_tolerance",
                "carry_heading_scale_deg", "lift_height"):
        if not math.isfinite(defaults[key]) or defaults[key] <= 0:
            raise ValueError("yaw.{} must be positive".format(key))
    for key in ("angular_speed_tolerance", "accuracy_weight", "braking_weight",
                "position_weight", "initial_yaw_range_deg", "target_offset_range_deg",
                "settled_linear_speed", "reference_loss_coef", "carry_heading_weight",
                "carry_angular_speed_weight", "in_place_min_yaw_deg"):
        if not math.isfinite(defaults[key]) or defaults[key] < 0:
            raise ValueError("yaw.{} must be nonnegative".format(key))
    for key in ("initial_yaw_range_deg", "target_offset_range_deg", "yaw_tolerance_deg"):
        if defaults[key] > 180:
            raise ValueError("yaw.{} must be at most 180".format(key))
    if not isinstance(defaults['staged_transport'], bool):
        raise ValueError('yaw.staged_transport must be boolean')
    probability = defaults['in_place_probability']
    if not isinstance(probability, (int, float)) or isinstance(probability, bool) or not 0 <= probability <= 1:
        raise ValueError('yaw.in_place_probability must be between zero and one')
    if probability > 0:
        if not defaults['enabled'] or not defaults['staged_transport']:
            raise ValueError('In-place sampling requires enabled staged yaw transport')
        if not defaults['yaw_tolerance_deg'] < defaults['in_place_min_yaw_deg'] <= defaults['target_offset_range_deg']:
            raise ValueError('In-place yaw minimum must exceed yaw tolerance and not exceed target offset range')
    return defaults


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quaternion_yaw(q):
    """Heading from an Isaac Gym quaternion in xyzw order."""
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quaternion(angle):
    zero = torch.zeros_like(angle)
    return torch.stack((zero, zero, torch.sin(angle / 2), torch.cos(angle / 2)), -1)


def yaw_features(error, angular_speed, arrived, speed_scale):
    return torch.stack((torch.sin(error), torch.cos(error), angular_speed / speed_scale,
                        arrived.to(error.dtype)), -1)


def update_yaw_phase(arrived, putdown, distance, error, angular_speed, config, lifted=None):
    """Arrival and putdown latch independently; rotation survives small position drift."""
    at_position = distance < config["position_tolerance"]
    if config['staged_transport']:
        if lifted is None:
            raise ValueError('Staged yaw transport requires pickup state')
        at_position = at_position & lifted
    arrived = arrived | at_position
    ready = (at_position & (error.abs() < math.radians(config["yaw_tolerance_deg"]))
             & (angular_speed.abs() < config["angular_speed_tolerance"]))
    return arrived, putdown | (arrived & ready)


def sample_in_place_targets(target_xy, object_xy, initial_yaw, target_yaw, config):
    """Replace a configurable fraction of translation goals with nontrivial turns."""
    if config['in_place_probability'] == 0:
        return target_xy, target_yaw, torch.zeros_like(initial_yaw, dtype=torch.bool)
    selected = torch.rand_like(initial_yaw) < config['in_place_probability']
    low = math.radians(config['in_place_min_yaw_deg'])
    high = math.radians(config['target_offset_range_deg'])
    magnitude = low + (high - low) * torch.rand_like(initial_yaw)
    sign = torch.where(torch.rand_like(initial_yaw) < 0.5, -1., 1.)
    yaw = torch.where(selected, initial_yaw + sign * magnitude, target_yaw)
    xy = torch.where(selected[:, None], object_xy, target_xy)
    return xy, yaw, selected


def update_pickup(lifted, reference_yaw, current_yaw, height, supported, upright,
                  rest_height, config):
    """Capture heading once, when all active carriers support an upright raised table."""
    picked_up = supported & upright & (height >= rest_height + config['lift_height'])
    first_pickup = ~lifted & picked_up
    return lifted | picked_up, torch.where(first_pickup, current_yaw, reference_yaw)


def control_heading_error(current_yaw, target_yaw, reference_yaw, arrived, config):
    """The same observation slots describe carry heading, then final heading."""
    reference = target_yaw
    if config['staged_transport']:
        reference = torch.where(arrived, target_yaw, reference_yaw)
    return wrap_angle(reference - current_yaw)


def transport_rotation_reward(error, angular_speed, lifted, arrived, config):
    """Penalties only during transport; no lingering reward for delaying arrival."""
    heading = (wrap_angle(error) / math.radians(config['carry_heading_scale_deg'])) ** 2
    speed = (angular_speed / config['angular_speed_scale']) ** 2
    penalty = config['carry_heading_weight'] * heading + config['carry_angular_speed_weight'] * speed
    return torch.where(lifted & ~arrived, -penalty, torch.zeros_like(penalty))


def rotation_reward(error, angular_speed, distance, arrived, putdown, supported, config):
    """Team pose reward after arrival, continuing through putdown and release.

    Hand support is required before putdown, but not once putdown has begun.
    """
    accuracy = 1.0 - wrap_angle(error).abs() / math.pi
    near_yaw = torch.exp(-(error / math.radians(config["braking_width_deg"])) ** 2)
    braking = near_yaw * (angular_speed / config["angular_speed_scale"]) ** 2
    position = torch.exp(-(distance / config["position_width"]) ** 2)
    reward = (config["accuracy_weight"] * accuracy - config["braking_weight"] * braking
              + config["position_weight"] * position)
    return torch.where(arrived & (supported | putdown), reward, torch.zeros_like(reward))


def placement_metrics(obj_states, target_pos, target_rot, hands_dist, active, putdown,
                      settled_count, config, rest_height=0.82):
    """Current settled placement, never the historical arrival latch alone."""
    distance = (obj_states[:, :2] - target_pos[:, :2]).norm(dim=-1)
    error = wrap_angle(quaternion_yaw(target_rot) - quaternion_yaw(obj_states[:, 3:7]))
    released = ((hands_dist > 0.07).all(-1) | ~active).all(-1)
    upright = 1 - 2 * (obj_states[:, 3].square() + obj_states[:, 4].square())
    resting_height = (obj_states[:, 2] - rest_height).abs() < config['settled_height_tolerance']
    valid = (putdown & released & resting_height & (upright > 0.95)
             & (distance < config["position_tolerance"])
             & (error.abs() < math.radians(config["yaw_tolerance_deg"]))
             & (obj_states[:, 7:10].norm(dim=-1) < config["settled_linear_speed"])
             & (obj_states[:, 10:13].norm(dim=-1) < config["angular_speed_tolerance"]))
    count = torch.where(valid, settled_count + 1, torch.zeros_like(settled_count))
    success = count >= config["settled_steps"]
    return distance, error, released, count, success
