# Joint heading-preserving transport and rotation-in-place

Use `cfg/teamhoi_ltable_yaw_transport.yaml` for one shared policy trained on all
four table shapes, with the same `[1,1,1,3]` asset weights: half L-table and half
split between round, square, and rectangular tables. Supply the assets in that
order. Formation, coverage, grasp sampling, assets, network dimensions, and the
original configuration files are unchanged.

## Task behavior

For each reset, sample the usual 3–10 m translation goal and initial/target yaw.
With probability `in_place_probability` (default 0.25, independently for every
shape/reset), replace the XY goal by the starting object XY. These in-place
requests sample a signed 10–30 degree yaw offset, avoiding an already-achieved
rotation goal. The other 75% retain the existing transport-goal sampling.

A pickup latch requires an upright object (local up dot world up > 0.95), object
root Z >= `obj_height + lift_height` (0.82 + 0.08 = 0.90 m), and both hands of
every active carrier within the existing 7 cm rim-proximity threshold. This uses
the existing contact proxy, not a measured force or a rigid grasp constraint.
Record object yaw once at that first qualifying pickup.

Before pickup, use the initial object yaw as the heading reference. During
transport, use the captured pickup yaw. After pickup AND reaching the existing
3 cm arrival radius, switch to the requested final yaw. In-place tasks cannot
enter arrival/putdown just because their initial XY already matches the goal;
they must first qualify as lifted. There is no prescribed carrying height or
hard orientation lock. These are reward and phase choices.

The goal encoder remains 7 inputs. Its four added yaw features remain sine,
cosine, yaw angular speed, and the arrival bit; only the heading reference is
phase-dependent in this new mode. Before arrival the angle describes heading
preservation; afterwards it describes the final rotation. Existing configs use
the old observation semantics.

## Reward changes

The new config sets `w[8] = 0`, disabling the native carrying alignment reward.
Other native reward weights stay unchanged. While lifted and before arrival:

```
carry_reward = -carry_heading_weight * (wrap(pickup_yaw - yaw) / radians(carry_heading_scale_deg))**2
               -carry_angular_speed_weight * (yaw_rate / angular_speed_scale)**2
```

Defaults are heading weight 1.0, yaw-speed weight 0.1, heading scale 30 degrees,
and angular-speed scale 1 rad/s. These are initial experimental settings, not
tuned coefficients. The penalty is zero before pickup and after arrival so it
does not oppose deliberate final rotation. Final yaw accuracy/braking and
putdown rewards continue as before. Existing task/discriminator reward mixing
still applies. These are RL reward terms, not gradients propagated through physics.

The old frozen-actor imitation coefficient is zero in this config. Its policy
uses different pre-arrival heading semantics and could oppose the new carrying
behavior. Other PPO, AMP and original motion rewards remain active.

Added per-environment `yaw_metrics` include pickup state, in-place episode flag,
carry heading error (radians), and transport rotation reward.

## Initialize a new experiment

Use the explicitly selected corrected familiar-object yaw checkpoint at
iteration **3,850**, not the L-table model and not the newer 5,050 checkpoint:

`/home/thomas/Documents/phd/code/CarryForge/checkpoints/teamhoi-yaw-corrected-3850.pth`

Verified SHA256:
`d84625ba05d49995727e51070f22c7129aa9f4b9a0f881b85dfb2798aaece437`.

 Use `--init_from_checkpoint`,
not `--checkpoint`: it copies actor/critic, discriminator and normalization
weights with a fresh optimizer, iteration count and environment state. Use a
new experiment name/output path. `--checkpoint` is reserved for resuming this
new experiment later. The existing seven-input checkpoint architecture is
compatible, but this behavior still needs fine-tuning.

The inherited budget is 2,000 updates (`max_epochs: 1999` under the native
trainer's stopping convention). No long training job is launched by installing
these changes. Select the environment count and minibatch/buffer sizes for the
available GPU; the local two-update smoke configuration is intentionally small
and is not the full training recipe.

For later CarryForge deployment, select this training config with the newly
trained checkpoint; old yaw checkpoints should retain their original config.
CarryForge's external waypoint setter also needs to rearm the new pickup state
when starting a fresh pickup leg. This change is currently scoped to training
and native episode resets, not deployment of an untrained behavior.

## Validation

CPU tests cover legacy behavior, pickup-before-in-place-rotation, pickup heading
capture, phase-dependent observations, continuity penalties, sampling ranges,
subset reset isolation, object mixture, and checkpoint architecture compatibility.
The native smoke check initializes pretrained weights, runs both PPO updates,
and checks finite model/optimizer tensors. Passing these checks establishes
implementation compatibility, not learned performance.


Example native training command (activate the existing workstation environment
and its library paths first):

```sh
cd /home/thomas/Documents/phd/code/TeamHOI-yaw
python teamhoi/run.py \
  --cfg_train cfg/teamhoi_ltable_yaw_transport.yaml \
  --init_from_checkpoint /home/thomas/Documents/phd/code/CarryForge/checkpoints/teamhoi-yaw-corrected-3850.pth \
  --assets default_round,default_square,default_rectangle,l_table_com \
  --system_max_humanoids 8 --min_humanoids 2 \
  --num_envs 128 --minibatch_size 4096 \
  --headless --sim_device cuda:0 --pipeline gpu --graphics_device_id -1 \
  --episode_length 900 --seed 42 \
  --exp_name TeamHOILTableTransportFrom3850 \
  --output_path output/transport-from-yaw3850
```

The example environment count is a starting allocation, not a measured throughput
or memory guarantee. Batch size is environments times horizon length; keep
minibatches compatible and `amp_minibatch_size <= minibatch_size` when scaling.
For two-carrier-only training add `--fix_num_humanoids 2`; otherwise native random
2–8-carrier training is retained.

Local validation on 2026-10-22: 27 CPU tests ran successfully (one optional
checkpoint test skipped). A 16-environment native smoke run initialized from
corrected yaw-3850 and completed two PPO updates, using eight L-table
environments and eight familiar-table environments, variable 2–8 active carriers,
and a sampled mix of transport and in-place episodes. Saved epoch was 2 and
maximum optimizer step was 4, confirming this was fresh initialization rather
than a training resume. Model and optimizer tensors were finite and model weights
changed. Logs: `/tmp/teamhoi-yaw-transport-from-3850-smoke.log`; temporary smoke
config: `/tmp/teamhoi-yaw-transport-smoke.yaml`. No full training run was started.
