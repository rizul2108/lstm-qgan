# LSTM-QGAN Run Instructions

All runnable code is inside `local_gpu/`. Run the commands below from the repository root unless a command first changes into `local_gpu`.

## 1. Enter the Runnable Folder

```powershell
cd local_gpu
```

## 2. Create a Virtual Environment

For GPU runs on Windows, use Python 3.12:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate
python -m pip install --upgrade pip
```

For CPU-only runs, any compatible Python environment can be used:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

## 3. Install Dependencies

### CPU Install

```powershell
python -m pip install -r requirements.txt
```

### GPU Install

Use this for NVIDIA GPU training/evaluation:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install pennylane pennylane-qiskit numpy scipy scikit-learn matplotlib Pillow tqdm
```

Or use the Windows helper:

```powershell
install.bat
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

For GPU mode, `torch.cuda.is_available()` must print `True`.

## 4. Run With CPU

CPU mode is useful for smoke tests and evaluation. Full training on CPU is very slow.

Small CPU training smoke test:

```powershell
python main.py --mode train --epochs 1 --subset 64 --batch_size 8 --loss bce --n_critic 1 --device cpu --output_dir outputs_cpu_smoke
```

CPU evaluation with an existing checkpoint:

```powershell
python main.py --mode evaluate --checkpoint outputs_long/generator_final.pth --output_dir outputs_cpu_eval --fid_samples 100 --device cpu
```

## 5. Run With GPU

Train on GPU:

```powershell
python main.py --mode train --epochs 45 --subset 5000 --batch_size 32 --loss wasserstein --n_critic 3 --lr 2e-4 --device cuda --output_dir outputs_long --resume --ckpt_interval 5
```

Evaluate on GPU:

```powershell
python main.py --mode evaluate --checkpoint outputs_long/generator_final.pth --output_dir outputs_long --fid_samples 500 --device cuda
```

Train and evaluate in one command:

```powershell
python main.py --mode both --epochs 45 --subset 5000 --batch_size 32 --loss wasserstein --n_critic 3 --lr 2e-4 --device cuda --output_dir outputs_long --fid_samples 500 --ckpt_interval 5
```

## 6. Resume Training

Training saves `checkpoint_latest.pth` inside the selected output folder. Resume with the same output folder and `--resume`:

```powershell
python main.py --mode train --epochs 45 --subset 5000 --batch_size 32 --loss wasserstein --n_critic 3 --lr 2e-4 --device cuda --output_dir outputs_long --resume --ckpt_interval 5
```

## 7. Useful Options

```text
--mode           train | evaluate | both      default: both
--loss           wasserstein | bce            default: wasserstein
--epochs         int                          default: 1000
--batch_size     int                          default: 128
--n_critic       int                          default: 5
--lr             float                        default: 2e-4
--latent_dim     int                          default: 64
--subset         int                          default: full MNIST train set
--output_dir     str                          default: outputs
--checkpoint     str                          default: output_dir/generator_final.pth
--device         cpu | cuda                   default: cpu
--fid_samples    int                          default: 500
--resume                                      resume from checkpoint_latest.pth
--ckpt_interval  int                          default: 10
```

## 8. Output Files

Output paths are relative to `local_gpu/`.

```text
outputs_long/generator_final.pth
outputs_long/discriminator_final.pth
outputs_long/checkpoint_latest.pth
outputs_long/generated_grid.png
outputs_long/generated_images/
outputs_long/plots/loss_wasserstein.png
outputs_long/plots/fid_per_class.png
```

## 9. IBM Quantum Notebook

Install optional IBM packages:

```powershell
python -m pip install pennylane-qiskit qiskit "qiskit-ibm-runtime>=0.20"
```

Then open:

```text
real_quantum_hardware.ipynb
```

The notebook expects trained weights at:

```text
local_gpu/outputs_long/generator_final.pth
```

## 10. CUDA Troubleshooting

If `--device cuda` fails with:

```text
AssertionError: Torch not compiled with CUDA enabled
```

then the active Python environment has CPU-only PyTorch. Reinstall CUDA PyTorch in the active environment:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

If pip says:

```text
ERROR: Could not find a version that satisfies the requirement torch
```

use Python 3.12 for the GPU environment:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
