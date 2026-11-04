"""Flag-free table pose commands. Pure Torch; quaternions are xyzw."""
import math
import torch

POSE_EXTRA_DIM = 9  # sin/cos of local roll,pitch,yaw error; local angular velocity


def pose_settings(config):
    values = dict(enabled=False, mode='yaw', walking_probability=.15,
                  angle_min_deg=5., angle_max_deg=30., initial_yaw_range_deg=5.,
                  rotation_weight=1., lift_height=.08, position_tolerance=.03,
                  angle_tolerance_deg=5., speed_tolerance=.1, settled_steps=15,
                  angular_speed_scale=1.)
    unknown = set(config) - set(values)
    if unknown:
        raise ValueError('Unknown pose_control settings: {}'.format(sorted(unknown)))
    values.update(config)
    if not isinstance(values['enabled'], bool) or values['mode'] not in ('yaw', 'pitch_roll'):
        raise ValueError('pose_control requires boolean enabled and yaw/pitch_roll mode')
    if not 0 <= values['walking_probability'] <= 1:
        raise ValueError('Invalid walking_probability')
    if not 0 < values['angle_min_deg'] <= values['angle_max_deg'] <= 90:
        raise ValueError('Pose angles must satisfy 0 < min <= max <= 90 degrees')
    for key in ('rotation_weight', 'lift_height', 'position_tolerance',
                'angle_tolerance_deg', 'speed_tolerance', 'settled_steps', 'angular_speed_scale'):
        if not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError('pose_control.{} must be positive'.format(key))
    if not 0 <= values['initial_yaw_range_deg'] <= 180:
        raise ValueError('Invalid initial yaw range')
    return values


def quat_mul(a, b):
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    return torch.cat((aw*bv + bw*av + torch.cross(av, bv, dim=-1),
                      aw*bw - (av*bv).sum(-1, keepdim=True)), -1)


def quat_conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def rotate_inverse(q, vector):
    v = -q[..., :3]
    t = 2 * torch.cross(v, vector, dim=-1)
    return vector + q[..., 3:] * t + torch.cross(v, t, dim=-1)


def axis_quaternion(angle, axis):
    result = torch.zeros(angle.shape + (4,), dtype=angle.dtype, device=angle.device)
    result[..., axis] = torch.sin(angle/2)
    result[..., 3] = torch.cos(angle/2)
    return result


def local_orientation_error(current, target):
    """Euler components of current^-1 * target, expressed about table-local axes.

    Commands contain a single local rotation, at most 30 degrees for tilt.
    Full quaternion targets are retained; this is not world Euler subtraction.
    """
    q = quat_mul(quat_conjugate(current), target)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x,y,z,w = q.unbind(-1)
    roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = torch.asin((2*(w*y-z*x)).clamp(-1., 1.))
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return torch.stack((roll, pitch, yaw), -1)


def pose_features(current, target, world_angular_velocity, speed_scale=1.):
    error = local_orientation_error(current, target)
    return torch.cat((torch.sin(error), torch.cos(error),
                      rotate_inverse(current, world_angular_velocity)/speed_scale), -1)


def sample_pose_targets(initial_rotation, initial_xy, walking_xy, config):
    n = initial_rotation.shape[0]
    walking = torch.rand(n, device=initial_rotation.device) < config['walking_probability']
    magnitude = math.radians(config['angle_min_deg']) + torch.rand(n, device=initial_rotation.device) * math.radians(config['angle_max_deg']-config['angle_min_deg'])
    angle = magnitude * torch.where(torch.rand_like(magnitude) < .5, -1., 1.)
    if config['mode'] == 'yaw':
        axes = torch.full((n,), 2, dtype=torch.long, device=initial_rotation.device)
    else:
        axes = torch.randint(0, 2, (n,), device=initial_rotation.device)
    angle = torch.where(walking, torch.zeros_like(angle), angle)
    local = torch.zeros_like(initial_rotation)
    local.scatter_(1, axes[:, None], torch.sin(angle/2)[:, None])
    local[:, 3] = torch.cos(angle/2)
    target = quat_mul(initial_rotation, local)
    xy = torch.where(walking[:, None], walking_xy, initial_xy)
    return xy, target, ~walking


def lifted_support(height, supported, rest_height, config):
    # Default object reset is rest_height + .03; avoid rewarding an unlifted table.
    return supported & (height >= rest_height + .03 + config['lift_height'])


def pose_rotation_reward(error, height, supported, rest_height, config):
    yaw_cost = error[..., 2].abs() / math.pi
    # Average the two tilt-axis errors to retain the agreed tilt reward scale.
    tilt_cost = error[..., :2].abs().mean(-1) / (math.pi/2)
    accuracy = 1. - yaw_cost - tilt_cost
    return torch.where(lifted_support(height, supported, rest_height, config),
                       config['rotation_weight']*accuracy, torch.zeros_like(accuracy))


def pose_hold_metrics(current, target, position, target_position, angular_velocity,
                      supported, rest_height, previous_count, config):
    relative = quat_mul(quat_conjugate(current), target)
    angle = 2*torch.atan2(relative[..., :3].norm(dim=-1), relative[..., 3].abs())
    distance = (position[..., :2]-target_position[..., :2]).norm(dim=-1)
    valid = (lifted_support(position[..., 2], supported, rest_height, config)
             & (angle < math.radians(config['angle_tolerance_deg']))
             & (distance < config['position_tolerance'])
             & (angular_velocity.norm(dim=-1) < config['speed_tolerance']))
    count = torch.where(valid, previous_count+1, torch.zeros_like(previous_count))
    return count, count >= config['settled_steps'], distance, angle
