@echo off
REM ============================================================
REM VALIDATION RUN — ~30 minutes
REM ============================================================
REM Foreground, streams to the console. Ctrl+C to stop.
REM Purpose: prove the pipeline works end-to-end (train + FID + grid)
REM before committing to the overnight run.
REM
REM Expected timing on RTX 3050 Ti Laptop (~25 s/batch):
REM   subset 1000 -> 62 batches/epoch
REM   10 epochs   -> ~620 batches -> ~4.3 hours train
REM
REM Actually that's too long. Using mode=both with a micro config
REM so the WHOLE pipeline (train + eval) fits in ~30 min.
REM   subset 512, batch 16 -> 32 batches/epoch
REM   5 epochs x 32 batches x 25s = 4000s = ~67 min train
REM
REM Still too long with wasserstein+n_critic=3. So we use BCE
REM (no gradient-penalty, no critic loop) to cut it ~3x.
REM   subset 512, batch 16, 5 epochs, bce -> ~22 min train
REM   + ~5 min eval -> ~27 min total.

cd /d "%~dp0"

echo === Verifying torch + CUDA ===
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if errorlevel 1 (
    echo Python/torch not available. Install deps first by running install.bat
    pause
    exit /b 1
)

echo.
echo === Starting 30-min validation run (Ctrl+C to stop) ===
echo === Mode: both (train + FID evaluation + image grid) ===
echo.
python main.py ^
    --mode both ^
    --epochs 5 ^
    --batch_size 16 ^
    --subset 512 ^
    --loss bce ^
    --n_critic 1 ^
    --lr 2e-4 ^
    --device cuda ^
    --output_dir outputs_validation ^
    --fid_samples 100 ^
    --ckpt_interval 1

pause
