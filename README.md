# LSTM-QGAN: Scalable NISQ Generative Adversarial Network

> Implementation of the paper:  
> **"LSTM-QGAN: Scalable NISQ Generative Adversarial Network"**  
> Cheng Chu, Aishwarya Hastak, Fan Chen — ICASSP 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Execution Flow](#execution-flow)
7. [Execution Steps](#execution-steps)
8. [Output Files](#output-files)
9. [Results](#results)
10. [IBM Quantum Hardware Validation](#ibm-quantum-hardware-validation)
11. [References](#references)

---

## Overview

LSTM-QGAN is a Quantum Generative Adversarial Network (QGAN) that:

- **Eliminates PCA preprocessing** — processes raw MNIST images directly.
- **Uses Quantum LSTM (QLSTM)** as a single scalable generator, instead of
  56 separate sub-generators as in PatchGAN.
- **Reduces hardware cost**: 5× fewer qubits, 5× fewer single-qubit gates,
  and 12× fewer two-qubit gates compared to the state-of-the-art PatchGAN.
- Achieves **lower FID scores** (193.28 vs 318.02 for PatchGAN on MNIST).

---

## Mathematical Formulation

### 1. GAN Objective (Equation 1)

The standard GAN minimax objective:

```
min_θg  max_θd  L{ D_θd(G_θg(z)),  D_θd(x) }
```

- **G(θ_g)**: Generator — maps noise z to fake images.
- **D(θ_d)**: Discriminator — scores real vs fake images.

### 2. Wasserstein Loss with Gradient Penalty (Equation 2)

```
min_θg  max_θd  E_{x~Pr}[D(x)]  −  E_{x̃~Pg}[D(x̃)]  −  λ · L_x̂

where:
    L_x̂ = E_{x̂~Px̂}[ (‖∇_{x̂} D(x̂)‖₂ − 1)² ]
    x̂   = ε·x + (1−ε)·x̃,   ε ~ Uniform[0,1]
    λ   = 10   (gradient penalty coefficient)
```

This is more stable than BCE loss and avoids mode collapse.

### 3. QLSTM Gate Equations

Each gate replaces the classical linear transformation W·[h,x] with a QNN:

```
f_t = σ( QNN_f([h_{t-1}, x_t]) )       ← forget gate
i_t = σ( QNN_i([h_{t-1}, x_t]) )       ← input  gate
g_t = tanh( QNN_g([h_{t-1}, x_t]) )    ← candidate memory
o_t = σ( QNN_o([h_{t-1}, x_t]) )       ← output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t       ← cell state update
h_t = o_t ⊙ tanh(c_t)                  ← hidden state update
```

### 4. QNN Circuit (7 qubits, 2 VQC layers)

```
For each layer l = 1, 2:
    Amplitude encoding of input
    For each qubit q:
        RX(θ_{l,q,0}) · RY(θ_{l,q,1}) · RZ(θ_{l,q,2})
    For each qubit q:
        CNOT(q, (q+1) mod 7)          ← ring entanglement
Measure: Pauli-Z expectation on all qubits → 7 values in [-1,1]
```

### 5. FID Score

```
FID = ‖μ_r − μ_g‖²  +  Tr(Σ_r + Σ_g − 2·sqrt(Σ_r · Σ_g))
```

Where μ, Σ are mean and covariance of Inception-v3 features.  
**Lower FID = better image quality.**

### 6. Patch-Based Image Generation

```
Patch size P = 196 pixels   (14 × 14)
Time steps T = 784 / 196  = 4
At each step t:  patch_t = G(z, h_{t-1})
Final image   = concat(patch_1, patch_2, patch_3, patch_4)
```

The generator produces a full 28×28 MNIST image sequentially, one 14×14 patch
per QLSTM time step.

---

## Project Structure

```
lstm-qgan-main/
│
├── main.py                          # Entry point — train / evaluate / both
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── real_quantum_hardware.ipynb      # IBM Quantum hardware validation notebook
├── sim_results.npy                  # Saved simulator results (pre-run)
├── real_results.npy                 # Saved real hardware results (pre-run)
├── real_hw_noise_analysis.png       # Hardware vs simulator comparison plot
│
├── src/                             # Core source modules
│   ├── __init__.py
│   ├── qnn_circuit.py               # 7-qubit variational quantum circuit (VQC)
│   ├── qlstm_cell.py                # QLSTM cell — 4 QNN gates replacing classical linear layers
│   ├── generator.py                 # Full LSTM-QGAN generator (2 stacked QLSTM layers)
│   ├── discriminator.py             # Classical MLP discriminator
│   ├── losses.py                    # Wasserstein + gradient penalty / BCE loss functions
│   ├── train.py                     # Training loop with checkpointing
│   ├── data.py                      # MNIST dataloader + patch utilities
│   └── evaluate.py                  # FID computation (Inception-v3) + image grid generation
│
└── local_gpu/                       # GPU training setup (Windows)
    ├── main.py                      # Identical to root main.py
    ├── requirements.txt             # Same dependencies
    ├── src/                         # Identical to root src/
    ├── install.bat                  # One-click dependency install
    ├── run_local_bg.bat             # Launch training as a background process
    ├── run_local.bat                # Interactive training launcher
    ├── run_local.ps1                # PowerShell training launcher
    ├── verify_quick.py              # Sanity-check script (model loads, forward pass works)
    └── outputs_long/                # Training outputs (50 epochs)
        ├── generator_final.pth      # Final generator weights  ← used for FID + IBM notebook
        ├── discriminator_final.pth  # Final discriminator weights
        ├── checkpoint_latest.pth    # Full resumable checkpoint (weights + optimizers + epoch)
        ├── generated_images/        # Sample grids saved every ckpt_interval epochs
        └── plots/                   # Loss curves (wasserstein / bce)
```

> **Note:** `local_gpu/main.py` and `local_gpu/src/` are identical to the root versions.
> The `local_gpu/` folder exists purely for convenience when training on a local Windows GPU
> machine, keeping outputs and launchers self-contained.

---

## Requirements

| Package        | Version   | Purpose                              |
|----------------|-----------|--------------------------------------|
| pennylane      | ≥ 0.35.0  | Quantum circuit simulation           |
| torch          | ≥ 2.0.0   | Deep learning framework              |
| torchvision    | ≥ 0.15.0  | MNIST dataset + Inception-v3 for FID |
| numpy          | ≥ 1.24.0  | Numerical computing                  |
| scipy          | ≥ 1.10.0  | Matrix square root for FID           |
| scikit-learn   | ≥ 1.2.0   | Utilities                            |
| matplotlib     | ≥ 3.7.0   | Plotting                             |
| tqdm           | ≥ 4.65.0  | Progress bars                        |

---

## Installation

### Step 1 — Clone / Download the project

```bash
git clone <your-repo-url>
cd lstm-qgan-main
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Linux / macOS:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

On Windows with a GPU, you can use the provided one-click installer instead:

```bash
cd local_gpu
install.bat
```

> **IBM Quantum hardware (optional):** To run `real_quantum_hardware.ipynb` on a real
> IBM backend, also install:
> ```bash
> pip install pennylane-qiskit qiskit "qiskit-ibm-runtime>=0.20"
> ```

---

## Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Install dependencies  (pip install -r requirements.txt) │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Train the model                                         │
│     python main.py --mode train --device cuda               │
│     Outputs: generator_final.pth, checkpoint_latest.pth,   │
│              generated_images/, plots/loss_*.png            │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Evaluate — FID scores + image grid                      │
│     python main.py --mode evaluate --device cuda            │
│     Outputs: plots/fid_per_class.png, generated_grid.png    │
└────────────────────────────┬────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  4. (Optional) IBM Quantum hardware validation              │
│     Open real_quantum_hardware.ipynb and run all cells      │
└─────────────────────────────────────────────────────────────┘
```

Each step is independent — if you already have a trained `generator_final.pth`
you can jump straight to step 3.

---

## Execution Steps

> ### ⚠️ Training Time Warning — GPU Required
>
> The generator uses PennyLane quantum circuits. Every generator forward pass runs
> **32 quantum circuits** (2 QLSTM layers × 4 cells × 4 QNN gates), which takes
> **~20–25 seconds per batch** even on a modern GPU. CPU training is not feasible
> for any meaningful number of epochs.
>
> Realistic timing with `--subset 5000 --batch_size 32` (~156 batches/epoch):
>
> | Hardware | Time per epoch | 50 epochs total |
> |----------|---------------|-----------------|
> | CPU      | several hours  | not feasible    |
> | GPU (e.g. RTX 3060) | ~50–65 min | ~2–3 days |
>
> Use `local_gpu/` with a CUDA-enabled GPU for all training.

---

### Step 1 — Train the Model

From the `local_gpu/` folder on Windows with a GPU:

```bash
cd local_gpu

# Run in background (recommended — keeps running if terminal closes):
run_local_bg.bat

# Or run interactively:
python main.py \
    --mode train \
    --epochs 50 \
    --subset 5000 \
    --loss wasserstein \
    --batch_size 32 \
    --device cuda \
    --output_dir outputs_long
```

Training saves a checkpoint every 10 epochs (`checkpoint_latest.pth`) so it can
be resumed if interrupted:

```bash
python main.py \
    --mode train \
    --epochs 50 \
    --subset 5000 \
    --batch_size 32 \
    --device cuda \
    --output_dir outputs_long \
    --resume
```

---

### Step 2 — Evaluate (FID Scores + Image Grid)

Once training completes and `generator_final.pth` is saved, run evaluation:

```bash
python main.py \
    --mode evaluate \
    --checkpoint local_gpu/outputs_long/generator_final.pth \
    --output_dir fid_results \
    --fid_samples 500 \
    --device cuda
```

This will:
- Generate a 10×10 grid of fake MNIST images → `fid_results/generated_grid.png`
- Compute per-class FID scores for digits 0–9 using Inception-v3 features
- Print average FID to terminal
- Save a bar chart → `fid_results/plots/fid_per_class.png`

> **FID evaluation timing:** Running 5,000 images through Inception-v3 takes
> ~3–8 minutes on GPU and ~20–40 minutes on CPU. Reduce `--fid_samples 100`
> for a faster but less statistically stable estimate.

---

### Step 3 — Train with BCE Loss (for comparison)

To reproduce the BCE vs Wasserstein loss comparison from Fig. 7 of the paper:

```bash
python main.py \
    --mode train \
    --loss bce \
    --epochs 50 \
    --subset 5000 \
    --batch_size 32 \
    --device cuda \
    --output_dir outputs_bce
```

---

### Step 4 — Train + Evaluate in One Step

```bash
python main.py \
    --mode both \
    --epochs 50 \
    --subset 5000 \
    --batch_size 32 \
    --device cuda \
    --output_dir outputs_long
```

---

### All Command-Line Options

```
--mode           train | evaluate | both      (default: both)
--loss           wasserstein | bce            (default: wasserstein)
--epochs         int                          (default: 1000)
--batch_size     int                          (default: 128)
--n_critic       int  (D steps per G step)   (default: 5)
--lr             float                        (default: 2e-4)
--latent_dim     int                          (default: 64)
--subset         int  (cap dataset size)      (default: None = full 60k)
--output_dir     str                          (default: outputs)
--checkpoint     str  (path to .pth file)     (default: None)
--device         cpu | cuda                   (default: cpu)
--fid_samples    int                          (default: 500)
--resume                                      (resume from checkpoint_latest.pth)
--ckpt_interval  int  (epochs between saves)  (default: 10)
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/generator_final.pth` | Final generator weights (use for evaluation + IBM notebook) |
| `outputs/discriminator_final.pth` | Final discriminator weights |
| `outputs/checkpoint_latest.pth` | Full resumable checkpoint (weights + optimizer states + epoch) |
| `outputs/generated_images/epoch_XXXX.png` | Sample image grids saved every `ckpt_interval` epochs |
| `outputs/plots/loss_wasserstein.png` | Generator & discriminator loss curves |
| `outputs/plots/fid_per_class.png` | FID score per digit class (bar chart) |
| `outputs/generated_grid.png` | Final 10×10 generated image grid |

---

## Results

### FID Scores (paper — MNIST, 1000 epochs)

| Model | Avg FID ↓ |
|-------|-----------|
| PatchGAN (baseline) | 318.02 |
| **LSTM-QGAN (ours)** | **193.28** |

### Hardware Cost Comparison

| Metric | PatchGAN | LSTM-QGAN | Reduction |
|--------|----------|-----------|-----------|
| Total qubits | 280 | 56 | **5×** |
| Single-qubit gates | 1680 | 336 | **5×** |
| Two-qubit gates | 1344 | 112 | **12×** |

> **Note on our training run:** Our implementation was trained for 50 epochs (vs 1000 in
> the paper) on a 5,000-sample MNIST subset, due to the significant time cost of simulating
> quantum circuits on classical hardware (~20–25s per batch on GPU). FID scores from our
> run will be higher than the paper's 193.28 — this is expected at reduced epochs/data.
> The IBM hardware validation and circuit behaviour remain fully reproducible.

---

## IBM Quantum Hardware Validation

The notebook `real_quantum_hardware.ipynb` validates that trained VQC parameters run
correctly on real IBM quantum hardware, by comparing Pauli-Z expectation values between
the PennyLane simulator and an actual IBM backend.

> **This is circuit validation, not image generation.** We extract the learned weights from
> one QLSTM gate and verify that the same gate behaviour is reproduced on real qubits under
> realistic hardware noise.

### Prerequisites

1. **IBM Quantum account** — free account at [quantum.ibm.com](https://quantum.ibm.com)
2. **API key** — copy from your IBM Quantum dashboard
3. **Trained weights** — `local_gpu/outputs_long/generator_final.pth` must exist
4. **Extra packages:**

```bash
pip install pennylane-qiskit qiskit "qiskit-ibm-runtime>=0.20"
```

### Steps to Run

**Step 1 — Ensure trained weights exist.** If you haven't trained yet:

```bash
cd local_gpu
run_local_bg.bat
```

**Step 2 — Open the notebook** in VS Code or Jupyter:

```
real_quantum_hardware.ipynb
```

**Step 3 — Set your IBM API key** in Cell 3:

```python
IBM_API_KEY = 'PASTE_YOUR_IBM_API_KEY_HERE'
```

**Step 4 — Run all cells in order** (Shift+Enter or "Run All"):

| Cell | What it does |
|------|-------------|
| Cell 1 | Install check |
| Cell 2 | Imports |
| Cell 3 | Configuration — set your API key here |
| Cell 4 | Load trained weights from `generator_final.pth` |
| Cell 5 | Run circuit on PennyLane simulator, collect reference results |
| Cell 6 | Build 16 fully-bound Qiskit circuits |
| Cell 7 | Connect to IBM Quantum, select least-busy backend |
| Cell 8 | Transpile circuits for the target backend |
| Cell 9 | Submit single job (all 16 circuits) |
| Cell 10 | Collect results and compute Pauli-Z expectation values |
| Cell 11 | Generate 3-panel comparison plot |
| Cell 12 | Results interpretation |

### Our Results (ibm_kingston, 156 qubits)

| Metric | Value |
|--------|-------|
| Backend | ibm_kingston (156 qubits) |
| Circuits submitted | 16 |
| Shots per circuit | 1024 |
| Overall MAE (sim vs real) | **0.088** |
| Per-qubit MAE range | 0.080 – 0.096 |

Simulator mean Pauli-Z per qubit:
```
[ 0.032, -0.037, -0.006, -0.007, -0.004, -0.028, -0.014]
```
Real hardware mean Pauli-Z per qubit:
```
[ 0.019,  0.009,  0.015,  0.010,  0.004, -0.001,  0.005]
```

An MAE of ~0.088 is within the expected noise range for IBM free-tier hardware (0.05–0.25).
Pre-computed result files `sim_results.npy` and `real_results.npy` are included so the
analysis plot can be reproduced without re-running the hardware job.

### Output Plot

The notebook saves `real_hw_noise_analysis.png` with three panels:

- **Scatter plot** — simulator vs real hardware Pauli-Z values per qubit per sample.
- **Per-qubit MAE bar chart** — mean absolute error per qubit.
- **Signed error heatmap** — which samples and qubits are most affected by noise.

### Important Notes

- The free IBM Open Plan does **not** support Sessions — the notebook uses
  `Sampler(mode=backend)` directly (no `Session` wrapper).
- All 16 circuits are batched into **one job** to minimise queue wait time.
- The job ID is printed so you can retrieve results later if the connection drops.
- `channel='ibm_quantum_platform'` is required for `qiskit-ibm-runtime >= 0.20` —
  the old `ibm_quantum` channel string is no longer valid.

---

## References

1. Cheng Chu et al., *"LSTM-QGAN: Scalable NISQ Generative Adversarial Network"*, ICASSP 2025.
2. He-Liang Huang et al., *"Experimental quantum generative adversarial networks for image generation"*, Physical Review Applied, 2021. (PatchGAN baseline)
3. Samuel Y. Chen et al., *"Quantum long short-term memory"*, ICASSP 2022. (QLSTM)
4. Gulrajani et al., *"Improved Training of Wasserstein GANs"*, NeurIPS 2017. (WGAN-GP)
5. IBM Quantum: https://quantum.ibm.com
