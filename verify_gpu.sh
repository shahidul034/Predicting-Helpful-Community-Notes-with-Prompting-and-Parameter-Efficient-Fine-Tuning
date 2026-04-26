#!/bin/bash
# GPU Verification Script
# Checks GPU availability and configuration

echo "=== GPU Configuration Check ==="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi

# List GPUs
echo "Available GPUs:"
nvidia-smi -L
echo ""

# Show detailed GPU info
echo "GPU Details:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader
echo ""

# Check CUDA visible devices
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-not set}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
echo ""

# Verify GPU 2 specifically (default for this project)
DEFAULT_GPU="2"
if [ "$CUDA_DEVICES" = "$DEFAULT_GPU" ]; then
    echo "GPU $DEFAULT_GPU is set as the visible device."
    nvidia-smi -i "$DEFAULT_GPU" --query-gpu=name,memory.total,memory.free --format=csv,noheader
elif [ "$CUDA_DEVICES" = "not set" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set."
    echo "Default GPU for this project: $DEFAULT_GPU"
    if nvidia-smi -i "$DEFAULT_GPU" --query-gpu=name --format=csv,noheader &> /dev/null; then
        echo "GPU $DEFAULT_GPU is available."
    else
        echo "WARNING: GPU $DEFAULT_GPU is not available."
    fi
else
    echo "Custom CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
fi

echo ""

# Check CUDA version
echo "CUDA Version:"
nvcc --version 2>/dev/null || nvidia-smi --query-gpu.driver_version --format=csv,noheader
echo ""

# Python GPU check
echo "Python GPU Check:"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "(Python/PyTorch not available)"

echo ""
echo "=== GPU Check Complete ==="
