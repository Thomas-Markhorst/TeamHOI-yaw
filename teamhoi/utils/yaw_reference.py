"""A frozen pretrained actor used only to regularize pre-arrival actions."""
import copy
import torch


def build_reference(agent, saved=None):
    agent._yaw_reference = copy.deepcopy(agent.model.a2c_network).eval()
    agent._yaw_reference.requires_grad_(False)
    agent._yaw_reference_norm = copy.deepcopy(agent.running_mean_std).eval()
    if saved is not None:
        agent._yaw_reference.load_state_dict(saved['model'], strict=True)
        agent._yaw_reference_norm.load_state_dict(saved['normalization'])


def reference_loss(agent, raw_obs, mask, mask_others, student_mu):
    # The last self/task input is the raw arrival bit, before teammate features.
    before_arrival = raw_obs[mask, agent.self_obs_size - 1] < 0.5
    if not before_arrival.any():
        return student_mu.sum() * 0.0
    with torch.no_grad():
        teacher_obs = raw_obs.clone()
        if agent.normalize_input:
            teacher_obs[mask, :agent.self_obs_size] = agent._yaw_reference_norm(
                teacher_obs[mask, :agent.self_obs_size])
        teacher_mu, _ = agent._yaw_reference.eval_actor(teacher_obs, mask, mask_others)
    return (student_mu[before_arrival] - teacher_mu[before_arrival]).square().mean()
