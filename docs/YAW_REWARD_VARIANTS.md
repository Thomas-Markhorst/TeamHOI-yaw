# Destination yaw reward comparison

Both configurations derive from cfg/teamhoi_yaw.yaml, preserving the original
adapted-yaw task, network, formation rewards, initial/target angle sampling,
reference-policy coefficient, putdown gates, and final-placement criteria.
They explicitly disable staged transport, carry-heading and angular-speed
penalties during transport, and in-place goal sampling. The yaw observation
therefore describes final target heading throughout the episode. Added yaw
rewards activate only after arrival and continue through putdown and release.

- cfg/teamhoi_yaw_only_error.yaml selects rotation_reward_only_error:
  r = accuracy_weight * (1 - abs(wrapped_yaw_error)/pi).
- cfg/teamhoi_yaw_only_error_drift.yaml selects rotation_reward_only_error_drift:
  r = accuracy_weight * (1 - abs(wrapped_yaw_error)/pi)
      + 0.3 * exp(-20 * distance_m^2).

This retains the author's corrected positive sign and distance scale of 20.
Both configs set position_weight=0.3; the error-only function ignores it.
Neither variant has a yaw-speed penalty, but the existing <0.1 rad/s putdown
readiness requirement is retained. The positional term is a reward for the
current distance to the goal, not a time difference of position or drift speed.
The existing transport reward remains in the original task reward.

The training task still calls rotation_reward(), which dispatches according to
config.yaw.reward_variant. Omitted reward_variant retains the original combined
accuracy/braking/position reward, so existing configurations keep their behavior.

Use the same source checkpoint, seed, assets, team sizes and training budget for
both experiments. For a comparison with the first adapted-yaw experiment, start
both from the same original TeamHOI checkpoint using --init_from_checkpoint.
Use distinct experiment names/output directories. Example inside the existing
DGX environment, substituting the source/output paths as needed:

```bash
python teamhoi/run.py --cfg_train cfg/teamhoi_yaw_only_error.yaml \
  --init_from_checkpoint /workspace/data/teamhoi-yaw/downloads/8agents.pth \
  --assets default_round,default_square,default_rectangle \
  --system_max_humanoids 8 --min_humanoids 2 --num_envs 1024 \
  --episode_length 900 --seed 42 --exp_name TeamHOIYawOnlyError --headless
```

For the second experiment, use cfg/teamhoi_yaw_only_error_drift.yaml and
--exp_name TeamHOIYawOnlyErrorDrift. Both configurations use a 4,000-iteration budget (`max_epochs: 3999` under
the existing stopping condition), save every 50 iterations, and retain numbered
checkpoints. Use distinct output directories for both runs.
