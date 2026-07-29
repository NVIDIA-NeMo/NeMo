# Prerequisites

Verify that your environment meets the following requirements before deploying the Nemotron Voicechat container.

## Hardware

The Nemotron Voicechat container requires an NVIDIA GPU with at least 80 GB of VRAM.

| GPU | Precision |
|-----|-----------|
| A100 | Mixed |
| H100 | Mixed |
| RTX 6000 Pro | Mixed |
| B200 | Mixed |

- CPU Architecture: x86_64 only.

## Operating System

Use a Linux distribution that meets the following requirements:

- Ubuntu 22.04 or later recommended.
- Supported by the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html).
- `glibc` >= 2.35. Verify with `ld -v`.

## CUDA Drivers

Install CUDA drivers by following the [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux).

- Use a [package manager installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation). Skip the CUDA toolkit -- the required libraries are bundled in the container.
- Install [open GPU kernel modules](https://github.com/NVIDIA/open-gpu-kernel-modules) matching your [driver version](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#driver-installation).

### Supported Driver Versions

| Major Version | EOL | Data Center and RTX/Quadro | GeForce |
|---|---|---|---|
| > 580 | TBD | Yes | Yes |

## Docker

Install Docker Engine for your Linux distribution by following the [Docker Engine installation guide](https://docs.docker.com/engine/install/).

After installation, verify that the Docker daemon is running and that your user can execute `docker` commands without `sudo`. Add your user to the `docker` group if needed:

```bash
sudo usermod -aG docker $USER
```

Log out and back in for the group change to take effect.

## NVIDIA Container Toolkit

The NVIDIA Container Toolkit enables Docker containers to access the host GPU.

1. Install the toolkit by following the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit).
2. Configure Docker to use the NVIDIA runtime by following the [Docker configuration steps](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).
3. Restart the Docker daemon after configuration:

   ```bash
   sudo systemctl restart docker
   ```

### Verify GPU Access

Confirm that containers can access the GPU:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

The output should display the driver version, CUDA version, and available GPU(s):

```text
| NVIDIA-SMI 585.01.07   Driver Version: 585.01.07   CUDA Version: 12.9     |
| GPU  Name                 ...
```

If this succeeds, your environment is ready to run the Nemotron Voicechat container.

---
Next: [Deploy and Run](deploy.md)
