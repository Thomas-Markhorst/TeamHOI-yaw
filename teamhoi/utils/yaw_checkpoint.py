"""Explicit, shape-checked initialization from an original TeamHOI checkpoint."""
import copy
import torch
from utils.yaw import LEGACY_GOAL_DIM, YAW_EXTRA_DIM


def expand_yaw_checkpoint(checkpoint, target_model, target_stats):
    """Return model/stats only. Optimizer and episode state are deliberately excluded."""
    source = checkpoint["model"]
    goal_keys = {"a2c_network.task_encoder.1.0.weight",
                 "a2c_network.task_encoder_critic.1.0.weight"}
    if set(source) != set(target_model):
        raise ValueError("Checkpoint model keys do not match this TeamHOI architecture")
    model = {}
    for key, target in target_model.items():
        value = source[key]
        if key in goal_keys:
            if (value.ndim != 2 or value.shape[1] != LEGACY_GOAL_DIM or
                    target.shape != (value.shape[0], LEGACY_GOAL_DIM + YAW_EXTRA_DIM)):
                raise ValueError("Expected original 3-input goal encoder at " + key)
            expanded = value.new_zeros(target.shape)
            expanded[:, :LEGACY_GOAL_DIM] = value
            model[key] = expanded
        else:
            if value.shape != target.shape:
                raise ValueError("Unexpected checkpoint shape at " + key)
            model[key] = value.clone()
    stats = copy.deepcopy(checkpoint["running_mean_std"])
    for key, fill in (("running_mean", 0.0), ("running_var", 1.0)):
        old = stats[key]
        if old.ndim != 1 or target_stats[key].shape != (old.numel() + YAW_EXTRA_DIM,):
            raise ValueError("Unexpected normalization shape at " + key)
        stats[key] = torch.cat((old, old.new_full((YAW_EXTRA_DIM,), fill)))
    return model, stats


def prepare_yaw_checkpoint(checkpoint, target_model, target_stats):
    """Initialize a new experiment from either legacy or already-yaw-aware weights."""
    source = checkpoint["model"]
    goal_key = "a2c_network.task_encoder.1.0.weight"
    if source[goal_key].shape[1] == LEGACY_GOAL_DIM:
        return expand_yaw_checkpoint(checkpoint, target_model, target_stats)
    if set(source) != set(target_model):
        raise ValueError("Checkpoint model keys do not match this TeamHOI architecture")
    if source[goal_key].shape[1] != LEGACY_GOAL_DIM + YAW_EXTRA_DIM:
        raise ValueError("Expected a 3-input or 7-input goal encoder")
    for key, target in target_model.items():
        if source[key].shape != target.shape:
            raise ValueError("Unexpected checkpoint shape at " + key)
    stats = checkpoint["running_mean_std"]
    if set(stats) != set(target_stats):
        raise ValueError("Checkpoint normalization keys do not match")
    for key, target in target_stats.items():
        if stats[key].shape != target.shape:
            raise ValueError("Unexpected normalization shape at " + key)
    return {key: value.clone() for key, value in source.items()}, copy.deepcopy(stats)


def initialize_from_checkpoint(agent, path):
    if not agent.vec_env.env.task.yaw_enabled:
        raise ValueError("--init_from_checkpoint requires the yaw task configuration")
    checkpoint = torch.load(path, map_location=agent.ppo_device)
    model, stats = prepare_yaw_checkpoint(checkpoint, agent.model.state_dict(),
                                         agent.running_mean_std.state_dict())
    # New experiments build their teacher from these selected weights, not the old saved teacher.
    agent._yaw_saved_reference = None
    agent.model.load_state_dict(model, strict=True)
    agent.running_mean_std.load_state_dict(stats)
    if "reward_mean_std" in checkpoint and agent.normalize_value:
        agent.value_mean_std.load_state_dict(checkpoint["reward_mean_std"])
    if agent._normalize_amp_input:
        agent._amp_input_mean_std.load_state_dict(checkpoint["amp_input_mean_std"])
        agent._amp_masked_input_mean_std.load_state_dict(checkpoint["amp_masked_input_mean_std"])
    print("Initialized yaw model from pretrained weights; optimizer and episode state are fresh.")
