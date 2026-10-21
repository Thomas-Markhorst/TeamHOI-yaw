FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.8 python3.8-dev python3-pip python-is-python3 git curl ca-certificates build-essential ninja-build libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 libvulkan1 vulkan-tools ffmpeg screen && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip==24.3.1 setuptools==69.5.1 wheel==0.45.1
RUN python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt /tmp/teamhoi-requirements.txt
RUN python -m pip install -r /tmp/teamhoi-requirements.txt scipy==1.10.1 PyYAML==6.0.2 Pillow==10.4.0 imageio==2.35.1 ninja==1.11.1.3 tensorboard==2.14.0 gym==0.23.1
COPY isaacgym-preview4.tar.gz /tmp/isaacgym-preview4.tar.gz
RUN tar -xzf /tmp/isaacgym-preview4.tar.gz -C /opt && python -m pip install --no-deps -e /opt/isaacgym/python && rm /tmp/isaacgym-preview4.tar.gz
RUN python -m pip install termcolor==2.4.0 matplotlib==3.7.5
RUN echo /opt/isaacgym/python > /usr/local/lib/python3.8/dist-packages/isaacgym.pth
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
WORKDIR /workspace/TeamHOI-yaw
CMD ["sleep", "infinity"]
