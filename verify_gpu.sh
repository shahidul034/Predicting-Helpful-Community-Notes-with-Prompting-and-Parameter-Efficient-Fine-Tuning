#!/bin/bash
# Verify GPU configuration

echo "=== GPU Configuration Check ==="

# Set GPU environment variables
export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"

echo ""
echo "=== NVIDIA GPU Info ==="
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "=== Python GPU Detection ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB')
" 2>/dev/null || echo "PyTorch not installed or CUDA not available"
