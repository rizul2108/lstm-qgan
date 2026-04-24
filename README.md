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
6. [Execution Steps](#execution-steps)
7. [Work Division](#work-division)
8. [Results](#results)
9. [IBM Quantum Hardware Validation](#ibm-quantum-hardware-validation)
10. [References](#references)

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

Each gate replaces classical linear transformation W·[h,x] with a QNN:

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

### 6. Patch Generation

```
Patch size P = 196 pixels   (14 × 14)
Time steps T = 784 / 196  = 4
At each step t:  patch_t = G(z, h_{t-1})
Final image   = concat(patch_1, patch_2, patch_3, patch_4)
```

---

## Project Structure

```
lstm-qgan-main/
│
├── main.py                          # Entry point – train / evaluate / both (Kaggle)
├── requirements.txt                 # Python dependencies (Kaggle)
├── README.md                        # This file
├── real_quantum_hardware.ipynb      # IBM Quantum hardware validation notebook
│
├── src/                             # Source modules (used by Kaggle / main.py)
│   ├── __init__.py
│   ├── qnn_circuit.py               # 7-qubit VQC
│   ├── qlstm_cell.py                # QLSTM cell with 4 QNN gates
│   ├── generator.py                 # Full LSTM-QGAN Generator
│   ├── discriminator.py             # Classical MLP Discriminator
│   ├── losses.py                    # Wasserstein + BCE losses
│   ├── train.py                     # Training loop
│   ├── data.py                      # MNIST loader + patch utilities
│   └── evaluate.py                  # FID + visual evaluation
│
└── local_gpu/                       # GPU training setup (Windows)
    ├── main.py                      # Same entry point
    ├── requirements.txt
    ├── install.bat                  # One-click dependency install
    ├── run_local_bg.bat             # Background training launcher
    ├── run_local.bat / .ps1         # Interactive training launchers
    ├── verify_quick.py              # Quick sanity-check script
    ├── src/                         # Same source modules (GPU-optimised)
    └── outputs_long/                # Trained model outputs (50 epochs)
        ├── generator_final.pth      # Final generator weights ← used by notebook
        ├── discriminator_final.pth
        ├── checkpoint_latest.pth    # Resumable checkpoint
        ├── generated_images/        # Sample grids saved during training
        └── plots/                   # Loss curves
```

---

## Requirements

| Package        | Version   | Purpose                        |
|----------------|-----------|--------------------------------|
| pennylane      | ≥ 0.35.0  | Quantum circuit simulation     |
| torch          | ≥ 2.0.0   | Deep learning framework        |
| torchvision    | ≥ 0.15.0  | MNIST dataset + Inception-v3   |
| numpy          | ≥ 1.24.0  | Numerical computing            |
| scipy          | ≥ 1.10.0  | Matrix square root for FID     |
| scikit-learn   | ≥ 1.2.0   | PCA utilities (baseline)       |
| matplotlib     | ≥ 3.7.0   | Plotting                       |
| tqdm           | ≥ 4.65.0  | Progress bars                  |

---

## Installation

### Step 1 — Clone / Download the project

```bash
# If using git:
git clone <your-repo-url>
cd LSTM_QGAN

# Or just navigate to the project folder:
cd LSTM_QGAN
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate:
# Linux / macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: PennyLane will use the `default.qubit` CPU simulator by default.
> To run on real IBM Quantum hardware, also install:
> ```bash
> pip install pennylane-qiskit
> ```
> and set up your IBM Quantum account credentials.

---

## Execution Steps

### Quick Smoke Test (recommended first run, ~5 minutes on CPU)

```bash
python main.py \
    --mode train \
    --epochs 100 \
    --subset 2000 \
    --loss wasserstein \
    --batch_size 32
```

### Full Training (as in the paper)

```bash
python main.py \
    --mode train \
    --epochs 1000 \
    --loss wasserstein \
    --batch_size 128 \
    --lr 2e-4
```

### Training with BCE Loss (for Fig. 7 comparison)

```bash
python main.py \
    --mode train \
    --loss bce \
    --epochs 1000 \
    --output_dir outputs_bce
```

### Evaluate a Saved Model

```bash
python main.py \
    --mode evaluate \
    --checkpoint outputs/generator_final.pth \
    --fid_samples 500
```

### Train + Evaluate in One Step

```bash
python main.py --mode both --epochs 1000
```

### All Command-Line Options

```
--mode          train | evaluate | both     (default: both)
--loss          wasserstein | bce           (default: wasserstein)
--epochs        int                         (default: 1000)
--batch_size    int                         (default: 128)
--lr            float                       (default: 2e-4)
--latent_dim    int                         (default: 64)
--subset        int  (cap dataset size)     (default: None = full 60k)
--output_dir    str                         (default: outputs)
--checkpoint    str  (path to .pth file)    (default: None)
--device        cpu | cuda                  (default: cpu)
--fid_samples   int                         (default: 500)
```

---

## Output Files

After training, you will find:

| File | Description |
|------|-------------|
| `outputs/generator_final.pth` | Saved generator weights |
| `outputs/discriminator_final.pth` | Saved discriminator weights |
| `outputs/generated_images/epoch_XXXX.png` | Sample images at each save interval |
| `outputs/plots/loss_wasserstein.png` | Loss curves (Fig. 7 of paper) |
| `outputs/plots/fid_per_class.png` | FID bar chart (Fig. 6b of paper) |
| `outputs/generated_grid_final.png` | Final 10×10 image grid (Fig. 6a) |

---
## Results (Expected)

Based on the paper (MNIST dataset, 1000 epochs):

| Model | Avg FID ↓ |
|-------|-----------|
| PatchGAN (baseline) | 318.02 |
| **LSTM-QGAN (ours)** | **193.28** |

Hardware cost comparison:

| Metric | PatchGAN | LSTM-QGAN | Reduction |
|--------|----------|-----------|-----------|
| Total Qubits | 280 | 56 | **5×** |
| Single-qubit gates | 1680 | 336 | **5×** |
| Two-qubit gates | 1344 | 112 | **12×** |

---

## IBM Quantum Hardware Validation

The notebook `real_quantum_hardware.ipynb` proves that the trained VQC parameters run correctly
on real IBM quantum hardware by comparing PauliZ expectation values between the PennyLane
simulator and an actual IBM backend.

> **This is circuit validation, not image generation.** We extract the learned weights from
> one QLSTM gate and verify that the same gate behaviour is reproduced on real qubits under
> realistic hardware noise.

### Prerequisites

1. **IBM Quantum account** — create a free account at [quantum.ibm.com](https://quantum.ibm.com)
2. **API key** — copy it from your IBM Quantum dashboard
3. **Trained weights** — run training first so `local_gpu/outputs_long/generator_final.pth` exists
4. **Python packages:**

```bash
pip install numpy torch matplotlib pennylane qiskit "qiskit-ibm-runtime>=0.20" scipy
```

### Steps to Run

**Step 1 — Train the model** (if not already done):

```bash
cd local_gpu
run_local_bg.bat        # Windows background training (50 epochs, ~several hours on GPU)
# or interactively:
python main.py --mode train --epochs 50 --subset 5000 --batch_size 32 --output_dir outputs_long
```

**Step 2 — Open the notebook** in VS Code (or Jupyter):

```
real_quantum_hardware.ipynb
```

**Step 3 — Set your IBM API key** in Cell 3 (Config):

```python
IBM_API_KEY = 'PASTE_YOUR_IBM_API_KEY_HERE'
```

**Step 4 — Run all cells in order** (Shift+Enter or "Run All"):

| Cell | What it does |
|------|-------------|
| Cell 1 | Install check (uncomment pip line if needed) |
| Cell 2 | Imports |
| Cell 3 | Configuration (set your API key here) |
| Cell 4 | Load trained weights from `generator_final.pth` |
| Cell 5 | Run circuit on PennyLane simulator, collect reference results |
| Cell 6 | Build 16 fully-bound Qiskit circuits |
| Cell 7 | Connect to IBM Quantum, select least-busy backend |
| Cell 8 | Transpile circuits for the target backend |
| Cell 9 | Submit single job (all 16 circuits, stays within 10-min budget) |
| Cell 10 | Collect results and compute PauliZ expectation values |
| Cell 11 | Generate 3-panel comparison plot |
| Cell 12 | Results interpretation |

### Actual Results (ibm_kingston, 156 qubits)

We ran the notebook on `ibm_kingston` (IBM free Open Plan) with 16 circuits × 1024 shots each.

| Metric | Value |
|--------|-------|
| Backend | ibm_kingston (156 qubits) |
| Circuits submitted | 16 |
| Shots per circuit | 1024 |
| Overall MAE (sim vs real) | **0.088** |
| Per-qubit MAE range | 0.080 – 0.096 |

Simulator mean PauliZ per qubit:
```
[ 0.032, -0.037, -0.006, -0.007, -0.004, -0.028, -0.014]
```
Real hardware mean PauliZ per qubit:
```
[ 0.019,  0.009,  0.015,  0.010,  0.004, -0.001,  0.005]
```

An MAE of ~0.088 is within the expected noise range for IBM free-tier hardware (0.05–0.25).
The saved result files `sim_results.npy` and `real_results.npy` are included in the repo.

### What Results to Expect When You Run

The notebook saves `real_hw_noise_analysis.png` with three panels:

- **Scatter plot** — simulator vs real hardware PauliZ values per qubit per sample.
- **Per-qubit MAE bar chart** — mean absolute error per qubit.
- **Signed error heatmap** — which samples and qubits are most affected by noise.

### Notes

- The free IBM Open Plan does **not** support Sessions — the notebook uses
  `Sampler(mode=backend)` directly (no `Session` wrapper).
- All 16 circuits are batched into **one job** to minimise queue wait time.
- Job ID is printed so you can retrieve results later if the connection drops.
- `channel='ibm_quantum_platform'` is required for `qiskit-ibm-runtime >= 0.20` — the old
  `ibm_quantum` channel and `ibm-q/open/main` instance format are no longer valid.

---

## References

1. Cheng Chu et al., *"LSTM-QGAN: Scalable NISQ Generative Adversarial Network"*, ICASSP 2025.
2. He-Liang Huang et al., *"Experimental quantum generative adversarial networks for image generation"*, Physical Review Applied, 2021. (PatchGAN)
3. Samuel Y. Chen et al., *"Quantum long short-term memory"*, ICASSP 2022. (QLSTM)
4. Gulrajani et al., *"Improved Training of Wasserstein GANs"*, NeurIPS 2017. (WGAN-GP)
5. IBM Quantum: https://quantum.ibm.com
