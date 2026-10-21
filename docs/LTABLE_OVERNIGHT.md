# COM-referenced L-table rotation experiment

Fine-tune the broad-angle-tested original-reward yaw checkpoint at iteration 3850.
Use the corrected pose reward already on this branch. Formation, coverage,
angular spacing, hand, lifting, transport and putdown rewards are unchanged.

The L asset and polygon sampler come from the existing CarryForge COM experiment.
The URDF and local grasp outline share its horizontal COM reference, with the
original vertical reference retained. All four shapes use 64 rim samples.
The new polygon span uses CarryForge's existing max-absolute-extent convention;
this experiment intentionally does not redesign formation or its coupling.

`cfg/teamhoi_ltable_yaw.yaml` assigns environments in asset order using weights
[1, 1, 1, 3]: round, square, rectangle, L. For 1024 environments this gives
171, 171, 170, 512. A single shared policy trains on the combined experience.
Initial yaw (+/-5 degrees) and target offset (+/-30 degrees) remain unchanged.
The frozen pre-arrival reference is freshly copied from the selected rotator,
with a uniform reference loss coefficient of 0.5, rather than the old 1.0.

`--init_from_checkpoint` now accepts original three-input or yaw-aware seven-input
goal checkpoints. Seven-input weights and normalization are copied exactly.
The source optimizer, epoch, environment state, and stored teacher are not
resumed. A new teacher comes from the selected source policy. Normal `--checkpoint`
resume continues to restore the experiment's saved teacher and optimizer.

From the repository inside the DGX container, substituting a new persistent
output directory and a frozen copy of the verified checkpoint:

```bash
python teamhoi/run.py \
  --cfg_train cfg/teamhoi_ltable_yaw.yaml \
  --init_from_checkpoint /workspace/data/teamhoi-yaw/ltable-overnight-01/source-03850.pth \
  --assets default_round,default_square,default_rectangle,l_table_com \
  --system_max_humanoids 8 --min_humanoids 2 --num_envs 1024 \
  --episode_length 900 --seed 42 --exp_name TeamHOILTable \
  --output_path /workspace/data/teamhoi-yaw/ltable-overnight-01/output --headless
```

The initial budget is 2000 PPO iterations (`max_epochs: 1999`, accounting for
this trainer's stopping condition). This is roughly an overnight budget at
previous throughput, not a guaranteed duration. The configuration saves numbered checkpoints every
50 iterations with `save_intermediate: True`, as in the previous DGX runs. Keep the exact source checkpoint hash,
code revision, command, and copied run config alongside outputs.

Before the long run, check CPU tests and a bounded four-shape simulation/training
smoke test, including polygon loading, fixed-joint collapse, finite observations
and rewards, fresh iteration state, and a reference matching the selected rotator.
Do not overwrite checkpoints or outputs from earlier runs.
