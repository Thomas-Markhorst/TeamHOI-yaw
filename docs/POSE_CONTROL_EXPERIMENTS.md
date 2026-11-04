# Four-agent pose control experiments

These are opt-in experiments. Original, destination-yaw, reward-comparison and L-table configurations keep their original behavior.

## Shared task

- Original `8agents.pth` initialization; fresh optimizer and environment state.
- Exactly four physical and active agents; preserve eight-agent transformer padding using `--system_max_humanoids 8 --fix_num_humanoids 4 --min_humanoids 4`.
- Default familiar round, square and rectangular tables. No L-table mixing in these experiments.
- Normal resets: agents start away from the resting table. They approach, grasp and lift.
- 15% level walking targets at the existing 3–10 m distances; retain initial heading.
- 85% rotation targets at the initial table XY; one local-axis command per episode. No transport leg in those cases.
- Initial table yaw remains uniform +/-5 degrees, as in the prior yaw comparison.
- Yaw run: one local yaw offset, uniform -30 to +30 degrees.
- Pitch/roll run: randomly choose one local roll OR pitch offset, magnitude 5–30 degrees, either sign. No combined commands or command switching yet.
- Targets are quaternion compositions `q_initial * q_local_command`. Orientation error is extracted from `inverse(q_current) * q_target`, not a difference of world Euler angles. Angular velocity is transformed into the current table frame. Errors use local roll/pitch/yaw Euler components, suitable for the bounded single-axis tilt commands; this is not a general SO(3) metric for arbitrary extreme tilts.

## Inputs and rewards

Both runs use the same 12-input goal: original XYZ plus sin(local error XYZ), cos(local error XYZ), and local angular velocity XYZ. There is no arrival bit. The existing goal token is enlarged; the attention architecture is unchanged. The vertical target is exposed, but there is no exact table-center-height tracking reward.

The orientation reward is:

    1 - abs(yaw_error)/pi
      - (abs(roll_error) + abs(pitch_error))/(pi/2)

This sums the three axis error costs while retaining one shared accuracy constant. A single-axis tilt has the requested pi/2 scale; yaw has the pi scale. Maximum is 1; every axis is supervised in both experiments, even when its target is zero. Rotation weight is 1 in the task reward. No angular deadband, velocity penalty, progress term or extra position-drift term.

Rotation reward is active only when all four agents have both hands within the existing 4 cm rim proximity threshold and table-root height exceeds the reset height by 8 cm. There is no arrival condition or latched rotation phase. The table resets at resting-height +3 cm, so the lifted threshold is approximately 0.93 m.

Keep original approach, formation, hand proximity/side/separation, contact and transport-position rewards and weights. Remove the equal-world-hand-height factor only for this task. Replace the fixed contacted-rim-height lifting reward with a contact-weighted lift fraction: clamp((table root z - reset z)/0.08, 0, 1). It rewards clearing the resting height and saturates; it does not require a precise carrying height. The transport-facing weight is zero. Formation and its multipliers are unchanged.

Keep putdown code/weight, but these commands never activate putdown. Arrival and putdown latches stay false; there are no post-putdown reward overrides. Explicit placement commands and command sequencing are future work.

Keep 0.5 task / 0.5 AMP mixture, PPO/value/action-bound losses, both AMP discriminators and penalties, and the original fixed action standard deviation. The frozen action-reference regularizer coefficient is zero. Both actor and critic goal input matrices gain nine zero-initialized columns; normalization gains zero means/unit variances. Existing pretrained behavior is retained at initialization except for the deliberately changed goal-height observation semantics.

## Success and termination

Success is a logging metric: full quaternion orientation error <5 degrees, XY distance <3 cm, angular speed <0.1 rad/s, all hands supported and table lifted, for 15 consecutive control steps. No exact carrying-height target, upright requirement, release requirement, completion bonus or success-triggered termination. Episodes continue holding until the normal time limit or failure. Original object-root height limits (about 0.70–1.30 m) and humanoid fall termination remain. Larger tilts may require a future review of those limits.

Training logs expose `Pose/*` metrics, sampled at the last step of each rollout, not episode success rates. A short smoke test validates execution, not learning or physical rotation success.

## Launch

Use separate outputs and seed 42, 1024 environments, episode length 900, rollout length 32. Configs stop after 4000 completed iterations and save numbered checkpoints every 50.

```bash
python teamhoi/run.py --task HumanoidAMPTableLift --headless \
  --cfg_train cfg/teamhoi_pose_yaw.yaml --cfg_env cfg/main.yaml \
  --num_envs 1024 --episode_length 900 \
  --system_max_humanoids 8 --fix_num_humanoids 4 --min_humanoids 4 \
  --seed 42 --init_from_checkpoint /path/to/original/8agents.pth \
  --output_path /path/to/unique/yaw/output
```

Use `cfg/teamhoi_pose_pitch_roll.yaml` and a separate output for the tilt run. Use the existing installation's usual CUDA, motion-data and tracking settings. Run CPU regression tests with:

```bash
TEAMHOI_BASE_CHECKPOINT=/path/to/original/8agents.pth OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```
