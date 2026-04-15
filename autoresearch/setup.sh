#!/bin/bash
# Setup script for currency prediction autoresearch
# Works on both CPU and GPU machines

set -e

echo "=== Currency Prediction Autoresearch Setup ==="

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# 2. Activate
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# 3. Install PyTorch (detect GPU)
python -c "import torch; print(f'torch already installed: {torch.__version__}')" 2>/dev/null || {
    echo "Installing PyTorch..."
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "  GPU detected, installing CUDA version..."
        pip install torch --index-url https://download.pytorch.org/whl/cu121
    else
        echo "  No GPU detected, installing CPU version..."
        pip install torch
    fi
}

# 4. Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# 5. Verify
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA:          {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:           {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:          {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')

import transformers
print(f'  Transformers:  {transformers.__version__}')

from transformers import Lfm2Model
print(f'  Lfm2Model:     available')

print()
print('Setup complete! Run:')
print('  python run_overnight.py          # baseline + 10 optimizer experiments')
print('  python run_optimizer.py --baseline-only  # just baseline')
print('  python -m pytest tests/ -v       # run tests')
"
