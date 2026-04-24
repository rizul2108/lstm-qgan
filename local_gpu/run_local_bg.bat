@echo off
setlocal EnableExtensions
REM ============================================================
REM CONTINUED RUN — subset 5000, batch 32, 50 epochs total
REM ============================================================
REM Launches training detached in a minimized window. Closing THIS
REM window does NOT kill training.
REM
REM Analysis after epoch 25:
REM   - Loss curves flatlined around epoch 5, no meaningful improvement
REM     between epochs 20-25. Images show blurry digit-like blobs (mostly
REM     8s/3s) with no sharpening in the final 5 epochs.
REM   - Resuming from epoch 25 checkpoint, targeting epoch 50 total.
REM     Expected ~25 more hours (~1 hr/epoch on RTX 3050 Ti).
REM
REM Tuned for RTX 3050 Ti Laptop (4 GB VRAM):
REM   subset 5000, batch 32  -> ~156 batches/epoch
REM   ~60 min/epoch (est.)
REM   50 epochs total -> ~25 h from now (resumable)
REM
REM   Images saved every 5 epochs so progress is visible sooner.
REM   Checkpoints saved every 5 epochs.
REM   --resume picks up from the latest checkpoint automatically.
REM   Just double-click this file again after any interruption.

cd /d "%~dp0"

set "OUT_DIR=outputs_long"
set "LOG_FILE=%OUT_DIR%\train.log"
set "ERR_FILE=%OUT_DIR%\train.err"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo.
echo === Launching continued run (subset=5000, batch=32, 50 epochs, lr=2e-4) ===
echo.

start "LSTM-QGAN long" /MIN cmd /v:on /k "python -u main.py --mode train --epochs 50 --batch_size 32 --subset 5000 --loss wasserstein --n_critic 3 --lr 2e-4 --device cuda --output_dir %OUT_DIR% --resume --ckpt_interval 5 1> %LOG_FILE% 2> %ERR_FILE% & set EXIT_CODE=!ERRORLEVEL!& echo.>> %LOG_FILE% & echo [LAUNCHER] Python exited with code !EXIT_CODE!.>> %LOG_FILE% & echo. & echo Python finished with exit code !EXIT_CODE!. & echo Logs: %LOG_FILE% and %ERR_FILE% & pause"

echo Training started in minimized window "LSTM-QGAN long".
echo Check logs at: %OUT_DIR%\train.log
echo.
pause
