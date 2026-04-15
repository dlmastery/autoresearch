@echo off
REM Setup script for currency prediction autoresearch (Windows)

echo === Currency Prediction Autoresearch Setup ===

REM 1. Create virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM 2. Activate
call venv\Scripts\activate

REM 3. Install PyTorch
python -c "import torch" 2>nul
if errorlevel 1 (
    echo Installing PyTorch...
    pip install torch --index-url https://download.pytorch.org/whl/cu121
)

REM 4. Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM 5. Verify
echo.
echo === Verification ===
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}'); import transformers; print(f'Transformers: {transformers.__version__}')"

echo.
echo Setup complete! Run:
echo   python run_overnight.py
echo   python run_optimizer.py --baseline-only
echo   python -m pytest tests/ -v
