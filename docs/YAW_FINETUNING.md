# Table yaw fine-tuning

Develop on ThomasWS; perform simulation smoke tests and training on DGX01.
Baseline: upstream sail-sg/TeamHOI commit 6fdc9885f8a9c82854be99adca6f2570e67ac63a.
This extension is opt-in through `cfg/teamhoi_yaw.yaml`; the original training config retains the original observation dimensions and task behavior.

## Task

The 3-value goal input becomes 7 values:

1. The existing three position/lowering values.
2. Sine and cosine of the signed shortest target-minus-current yaw error.
3. Table world-z angular velocity divided by `angular_speed_scale` (upright-table approximation).
4. A raw arrival bit, shared by the team.

Exact heading is modulo 360 degrees: 180 degrees is an error even for a symmetric table. Object geometry and agent-relative observations remain available to interpret the common turning objective.

Arrival uses the original 3 cm condition. It latches until reset. Only then is the shared rotation reward active, provided the existing all-active-agents carrying-contact condition holds. A small translation during rotation does not return to the transport stage. The putdown latch additionally requires current position within 3 cm, yaw error below 5 degrees, and table angular speed below 0.1 rad/s. There is no separate phase token: the arrival bit and the existing lowering cue describe the stage.

The added reward combines linear yaw accuracy, angular-speed damping that increases near the desired yaw, and local position accuracy. Existing support rewards remain active until putdown. The original putdown reward is then used unchanged; rotation reward stops. Final-placement evaluation still checks position, yaw, release, and consecutive settling, so the historical latch does not establish success. This first version does not implement recovery after putdown starts.

Initial yaw is sampled independently within +/-5 degrees. Target offset relative to that initial yaw is sampled within +/-30 degrees. Broaden the ranges in the config as learning improves; no automatic curriculum is claimed. Humanoids keep the original distant initialization.

## Initializing a NEW experiment (DGX01)

Use the same Isaac Gym / Python / CUDA-compatible environment as the upstream baseline. Ensure the original checkpoint is a real downloaded tensor file, not its Git LFS pointer, and that the reference motions required by `cfg_motions` are installed. Do not copy workstation-specific environment paths to DGX01.

Run from the repository root, replacing `/data/teamhoi/8agents.pth` with the persistent checkpoint path on DGX01:

```bash
python teamhoi/run.py \
  --cfg_train cfg/teamhoi_yaw.yaml \
  --init_from_checkpoint /data/teamhoi/8agents.pth \
  --system_max_humanoids 8 --min_humanoids 2 --num_envs 1024 \
  --assets default_round,default_square,default_rectangle \
  --episode_length 900 --exp_name TeamHOIYaw --headless
```

The 900-step episode is a starting budget for transport plus rotation, not a measured training requirement. Match `system_max_humanoids` to the source checkpoint. Keep seed, commit, configuration, machine, and checkpoint provenance with runs. Start with a bounded DGX smoke test before a long job; no convergence or DGX compatibility has yet been established.

`--init_from_checkpoint` copies pretrained model weights, expands the actor and critic's goal-input matrices from 3 to 7 columns, and zeros only the new columns. Existing normalization and both AMP normalizations are preserved; the four added coordinates start with zero mean and unit variance. Their statistics share the inherited running count. The action mapping initially matches the original model for matching old inputs.

The optimizer, epoch, and environment state are fresh. Do not pass an original checkpoint through `--checkpoint` with a yaw config: that is the strict resume path, not conversion.

A frozen copy of the initialized policy and its normalization regularizes actions before arrival. This reduces regression; it does not guarantee unchanged walking. The reference and its normalization are saved with the new checkpoint and restored on resume. It is not used for action selection at inference. AMP discriminator definitions and inputs remain unchanged; the existing discriminator updates continue during training.

## Resume

Use the same yaw configuration and explicitly pass the new yaw checkpoint:

```bash
python teamhoi/run.py --cfg_train cfg/teamhoi_yaw.yaml \
  --checkpoint output/TeamHOIYaw/ckpt.pth \
  --system_max_humanoids 8 --min_humanoids 2 --num_envs 1024 \
  --episode_length 900 --exp_name TeamHOIYaw --headless
```

Omit `--init_from_checkpoint` when resuming. Resume restores the new optimizer and frozen reference. The default output directory also supports automatic detection of that experiment's checkpoint. Use distinct experiment names for distinct configurations.

## CPU validation (ThomasWS or DGX01)

```bash
TEAMHOI_BASE_CHECKPOINT=/data/teamhoi/8agents.pth \
  python -m unittest discover -s tests -v
```

Without that variable the pure task-math and checkpoint-shape checks still run, and real-checkpoint parity is explicitly skipped. Parity exercises actor and critic outputs at 2, 4, and 8 agents, active-agent masks, the expanded observation layout, and the frozen reference. It does not establish policy quality or simulation correctness.

## Bounded evaluation (DGX01)

```bash
python teamhoi/run.py --cfg_train cfg/teamhoi_yaw.yaml \
  --checkpoint output/TeamHOIYaw/ckpt.pth --test --headless \
  --system_max_humanoids 8 --fix_num_humanoids 4 --num_envs 12 \
  --episode_length 900 --seed 42 \
  --yaw_eval_episodes 120 --yaw_eval_output output/eval-yaw-4agents.json
```

Reports include terminal pose errors, release, arrival and putdown latches, termination, and final settled success, grouped by shape and team size. Success requires all active agents to release both hands and correct final position/yaw, upright table at resting height, and low object linear/angular velocity for 15 consecutive control steps. It is measured at episode end, not latched at the first successful instant. Repeat across team sizes, yaw ranges and seeds, with a zero-offset condition to assess transport regression. A physics failure cannot count as success.

For pre-training DGX smoke tests only, create an inference-only converted checkpoint:

```bash
python scripts/convert_yaw_checkpoint.py /data/teamhoi/8agents.pth output/yaw-untrained.pth
```

Use it with the bounded evaluation command above, replacing `--checkpoint`. It has not learned rotation and is not a training-resume checkpoint. The tool refuses to overwrite an existing file.

## Files

- `teamhoi/env/tasks/humanoid_multi_amp_teamhoi_task.py`: sampling, observations, stage conditions, rewards, metrics.
- `teamhoi/utils/yaw.py`: simulator-independent math and conditions.
- `teamhoi/utils/yaw_checkpoint.py`: explicit pretrained initialization.
- `teamhoi/utils/yaw_reference.py`: frozen pre-arrival reference loss.
- `teamhoi/learning/transformer/trans_agent.py`: initialization, preservation loss, reference persistence, diagnostics.
- `teamhoi/learning/transformer/trans_players.py`: bounded evaluation and masked-AMP statistics restoration.
- `teamhoi/utils/yaw_evaluation.py`: terminal evaluation report.
- `teamhoi/utils/config.py`: initialization and evaluation flags, output-path-aware resume detection.

The existing goal encoder automatically uses its declared input dimension, so no attention/backbone modification is required. A related subset-indexing issue in the lowering cue is fixed, and inference now restores masked AMP normalization from its own saved statistics.
