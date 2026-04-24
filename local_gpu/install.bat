@echo off
REM One-shot installer. Run this once before the first training run.
REM Installs torch with CUDA support + PennyLane + other deps.
REM
REM NOTE on CUDA versions:
REM   nvidia-smi shows the DRIVER's max supported CUDA version.
REM   PyTorch wheels ship their own bundled CUDA runtime. A cu124 wheel runs
REM   fine on systems where nvidia-smi shows 12.4, 12.8, 13.0, etc. — as long
REM   as the driver is new enough. You do NOT need CUDA 13.0 toolkit installed.

cd /d "%~dp0"

echo === Installing dependencies ===
echo.
echo Installing PyTorch with cu124 wheel (works on CUDA driver 12.4 or newer,
echo including 13.0). If install fails, try cu121 by editing this script.
echo.

REM Uninstall any CPU-only torch that may already be present.
pip uninstall -y torch torchvision torchaudio 2>nul

REM PyTorch with CUDA 12.4 runtime — broadly compatible with modern drivers.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

REM Everything else.
pip install "pennylane>=0.35.0" numpy scipy scikit-learn matplotlib Pillow tqdm

echo.
echo === Verifying install ===
python -c "import torch; print('torch:', torch.__version__); print('CUDA compiled for:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY - check your driver')"
python -c "import pennylane; print('pennylane:', pennylane.__version__)"

echo.
echo Done. If CUDA available == True, run run_local_bg.bat to start training.
echo If CUDA available == False, paste the output above and ask for help.
pause

