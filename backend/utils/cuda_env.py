"""
A pose model on the CPU must never initialise CUDA

With the .onnx on the CPU the app died every 5-15 minutes with 0xC0000005
(once 0xC0000409), on an i7-7820HQ + Quadro M1200 with the NVIDIA driver
installed. What was measured:

- The Windows event log names nvdxgdmal64.dll_unloaded at offset 0x83b0 almost
  every time. A few entries name torchvision\\jpeg8.dll or httptools at fixed
  offsets: the same stale address, after another DLL was loaded into the freed
  range. The crashes fall on a 5-minute wall-clock grid, so something on the
  machine calls back into the process periodically.
- faulthandler showed every Python thread idle; the faulting thread is native.
- The .pt on the GPU ran for over an hour without a crash. onnxruntime 1.20.1
  instead of 1.23.2 did not help.
- Process Monitor shows who loads nvdxgdmal64.dll: c10_cuda.dll ->
  cudart64_110.dll (cudaGetDeviceCount) -> nvcuda64.dll -> LoadLibrary. That
  is PyTorch asking how many GPUs there are. The NVIDIA driver starts, sees no
  visible GPU, unloads that component, and something it left behind is called
  later.
- Patching torch.cuda.is_available and torch.cuda.device_count to return
  False / 0 before anything else ran kept nvdxgdmal64.dll from loading
  (Process Monitor) and the app ran stable.

Who asks, on device "cpu" (ultralytics 8.3.40, torch 2.5.1): select_device
sets CUDA_VISIBLE_DEVICES=-1 but short-circuits before asking torch. The call
is in nn/autobackend.py, when the predictor sets the model up on the first
predict: `torch.cuda.is_available() and device.type != "cpu"` - CUDA is asked
first, whatever the device. torch.cuda.device_count goes through NVML and
then cudaGetDeviceCount, so it is replaced too. Nothing else on the CPU predict
path calls into CUDA: ops.Profile only synchronises a cuda device,
AutoBackend.warmup is skipped on the CPU, and torchvision's CUDA version check
is a compile-time constant.

keep_cuda_off() replaces both for the rest of the process. It is applied only
when the pose device resolves to "cpu"; a GPU index needs the real thing.
"""

from __future__ import annotations

_kept_off = False


def _not_available() -> bool:
    return False


def _no_devices() -> int:
    return 0


def keep_cuda_off() -> None:
    """Make torch report no CUDA for the rest of the process, without asking it.

    Call before Ultralytics loads a model. Importing torch does not initialise
    CUDA; asking whether it is available does.
    """
    global _kept_off
    import torch

    torch.cuda.is_available = _not_available
    torch.cuda.device_count = _no_devices
    _kept_off = True


def cuda_kept_off() -> bool:
    """Whether keep_cuda_off() has been applied in this process."""
    return _kept_off
