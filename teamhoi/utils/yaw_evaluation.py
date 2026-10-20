"""Bounded evaluation using terminal placement, not historical arrival."""
import json
import math
from pathlib import Path
import torch


def evaluate_player(player, episodes, output_path):
    task = player.env.task
    if not task.yaw_enabled or episodes <= 0:
        raise ValueError('Yaw evaluation requires an enabled yaw task and positive episode count')
    records = []
    obs = player.env_reset()
    with torch.no_grad():
        while len(records) < episodes:
            actions = player.get_action(obs, is_determenistic=True)
            mask = obs['mask'].view(-1)
            full = actions.new_zeros((mask.numel(), actions.shape[-1]))
            full[mask] = actions
            _, _, done, info = player.env_step(player.env, full)
            metrics = info['yaw_metrics']
            done_ids = done.nonzero(as_tuple=False).flatten()
            for env_id in done_ids.tolist():
                if len(records) >= episodes:
                    break
                asset_id = next(i for i, (a, b) in enumerate(task.ranges) if a <= env_id < b)
                terminated = bool(info['terminate'][env_id].item())
                records.append(dict(
                    asset=task.assets[asset_id], team_size=int(task.mask[env_id].sum().item()),
                    steps=int(task.progress_buf[env_id].item()), terminated=terminated,
                    arrived=bool(metrics['arrived'][env_id].item()),
                    putdown=bool(metrics['putdown'][env_id].item()),
                    released=bool(metrics['released'][env_id].item()),
                    position_error_m=float(metrics['position_error'][env_id].item()),
                    yaw_error_deg=math.degrees(metrics['yaw_error'][env_id].item()),
                    angular_speed_rad_s=float(metrics['angular_speed'][env_id].item()),
                    success=bool(metrics['settled_success'][env_id].item()) and not terminated,
                ))
            # Resets only completed environments; empty IDs preserve active episodes.
            obs = player.env_reset(done_ids)
    groups = {}
    for row in records:
        key = '{} / {} agents'.format(row['asset'], row['team_size'])
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, rows in groups.items():
        summary[key] = dict(episodes=len(rows), success_rate=sum(r['success'] for r in rows) / len(rows),
                            mean_final_yaw_error_deg=sum(r['yaw_error_deg'] for r in rows) / len(rows),
                            mean_final_position_error_m=sum(r['position_error_m'] for r in rows) / len(rows))
    report = dict(checkpoint=task.cfg['args'].checkpoint, seed=task.cfg['seed'],
                  yaw_config=task.yaw_config, summary=summary, episodes=records)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Saved {} completed yaw evaluation episodes to {}'.format(len(records), path))
    return report
