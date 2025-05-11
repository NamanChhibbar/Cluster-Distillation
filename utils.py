import subprocess

import numpy as np
import torch

def gpu_usage() -> list[int]:
  '''
  Get the current GPU memory usage.
  '''
  # Get output from nvidia-smi
  try:
    result = subprocess.check_output([
      'nvidia-smi',
      '--query-gpu=memory.used',
      '--format=csv,nounits,noheader'
    ]).decode('utf-8').strip()
  except:
    print('nvidia-smi command not found. Ensure you have NVIDIA drivers installed.')
    return []
  # Extract memory used by GPUs in MiB
  gpu_memory = [int(mem) for mem in result.split('\n')]
  return gpu_memory

def get_device(threshold: int | float = 500) -> str:
  '''
  Returns a device with memory usage below `threshold`.
  '''
  # Check if CUDA is available
  if torch.cuda.is_available():
    usage = gpu_usage()
    cuda_ind = np.argmin(usage)
    return f'cuda:{cuda_ind}' if usage[cuda_ind] < threshold else 'cpu'
  # Check if MPS is available
  if torch.backends.mps.is_available():
    usage = torch.mps.driver_allocated_memory() / 1e6
    return 'mps' if usage < threshold else 'cpu'
  return 'cpu'
