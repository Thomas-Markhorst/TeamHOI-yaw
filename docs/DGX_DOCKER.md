# DGX01 container

Repository: /fs01/tud/tcmarkhorst/projects/TeamHOI-yaw
Branch: codex/yaw-finetune (initial deployment: 94f1674)
Image: teamhoi-yaw:latest
Container: teamhoi-yaw-container

## Enter

```bash
docker exec -it -w /workspace/TeamHOI-yaw teamhoi-yaw-container bash
```

Code is mounted read/write at /workspace/TeamHOI-yaw.
Persistent shared data is mounted at /workspace/data.
The container starts idle and exposes all GPUs, matching MotionSmith. Select
an available GPU explicitly for each job; visibility does not reserve GPUs.
No display socket or network ports are exposed.

## Rebuild

The image uses Python 3.8, CUDA 11.8, PyTorch 2.0.1, and Isaac Gym Preview 4.
The proprietary NVIDIA archive stays outside Git. Prepare a small build context:

```bash
context_dir=$(mktemp -d)
cp Dockerfile requirements.txt "$context_dir/"
cp /projB/tud/tcmarkhorst/data/teamhoi-yaw/downloads/isaacgym-preview4.tar.gz "$context_dir/"
docker build -t teamhoi-yaw:latest "$context_dir"
```

The original archive can be downloaded from https://developer.nvidia.com/isaac-gym-preview-4.
Start the container with `bash scripts/start_dgx_container.sh` from the host repo.
It starts the existing container if present, otherwise creates it.
After rebuilding the image, explicitly remove the idle old container before
starting it again to use the new image. Never remove it while jobs are running.
An optional docker-compose.yml is included for systems with Compose installed;
DGX01 currently uses the direct Docker script.

## Baseline weights

The real baseline checkpoint is stored persistently under
/workspace/data/teamhoi-yaw/downloads/8agents.pth. The version in Git may be
only an LFS pointer. Expected SHA256:
afc9853bbb58681599b3b6c0af3bdcf772b99a1a3cce8452ef96231fd7206d76.
Use the persistent path for --init_from_checkpoint.

Follow YAW_FINETUNING.md for later simulation and training. Container setup
does not establish that simulation or training succeeds. No experiment is
started by the container entrypoint.
