# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project overview

**STAT 238 Final Project — Stephen Lu (McGill)**

Goal: implement the *Adaptive Monte Carlo augmented with normalizing flows* algorithm (Gabrié et al. 2022, [arXiv:2105.12603](https://arxiv.org/abs/2105.12603)) and benchmark it against a naive MALA baseline. The target distribution is the Boltzmann distribution of **alanine dipeptide**, a 22-atom molecule used as a standard molecular-simulation benchmark.

The core idea: run `n_chains` parallel MCMC chains with local MALA steps, and every `lag` steps attempt a non-local jump from a normalizing flow `q` trained online (forward KL) purely on the chain samples. As the flow learns the target, its Metropolis acceptance rate rises and mixing improves.

---

## Python environment

Always use the project venv:

```bash
~/.venvs/310/bin/python
```

Do **not** use system `python` or `python3`; they lack PyTorch and the ML dependencies.

---

## Running the experiment

From the repo root:

```bash
~/.venvs/310/bin/python notebooks/main.py
```

This runs:
1. **MALA-only baseline** — 500 iterations × 20 steps × 100 chains  
2. **Adaptive MALA + Flow** — same budget, with a RealNVP trained online  
3. Saves `notebooks/history.pkl` (all logged metrics) and `notebooks/comparison.png` (6-panel figure)

To run a quick smoke test (tiny settings), use `importlib.util.spec_from_file_location` to load `notebooks/main.py` as a module and call `train_adaptive` with `n_iter=3, n_steps=3` on 4 chains.

---

## Repository layout

```
notebooks/
  main.py        ← ALL new implementation lives here (the script to run/debug)
  main.ipynb     ← Original notebook; contains dataset loading that main.py replicates
external/
  bgflow/        ← bgflow library (coordinate transforms, OpenMM energy wrapper)
  bgmol/         ← bgmol library (Ala2TSF300 dataset, system definition)
  flonaco/       ← flonaco library (reference RealNVP, sampling, training code)
data/
  Ala2TSF300.npy ← 1M-frame MD trajectory (downloaded automatically if missing)
src/             ← Earlier homework scripts (hw1–hw4), unrelated to the final project
outputs/         ← Saved figures from earlier homeworks
```

`external/` libraries are **not pip-installed**; they are inserted into `sys.path` at the top of `main.py`.

---

## Architecture of `notebooks/main.py`

The file is structured in execution order (module-level code runs on import):

### 1. Dataset & coordinate transform (module-level, always runs)

- Loads `Ala2TSF300` from `bgmol`: 1M frames of 66-dim Cartesian coordinates  
- Builds `bg.MixedCoordinateTransformation` (fit to the full dataset) which maps:
  - **Forward** (Cartesian → IC): `coordinate_transform.forward(x)` → `(bonds, angles, torsions, z_cart, dlogp)`
  - **Inverse** (IC → Cartesian): `coordinate_transform.forward(bonds, angles, torsions, z_cart, inverse=True)` → `(x, dlogp)`
- The mixed IC vector is `z = cat([bonds(17), angles(17), torsions(17), z_cart(9)])`, **dim=60**
- bonds/angles/torsions are normalized to `[0, 1]`; z_cart is whitened backbone (unconstrained)
- `target_energy = dataset.get_energy_model(n_workers=os.cpu_count())` — OpenMM energy in **kBT units** (β=1 implicit)

### 2. `MLP` / `ResidualAffineCoupling` / `FlowIC` — normalizing flow

Adapted verbatim from `external/flonaco/flonaco/{models.py, real_nvp_mlp.py}`.

- `FlowIC(dim=60)` — RealNVP operating in 60-dim IC space
  - `.forward(z_prior)` → `(z_IC, log_det_jac)` — **generative** direction (prior → data)  
  - `.backward(z_IC)` → `(z_prior, log_det_jac)` — **normalizing** direction (data → prior)  
  - `.nll(z_IC)` → `(batch,)` — per-sample negative log-likelihood  
  - `.sample(n)` → `(n, 60)` — IC samples drawn from `q`
- Default architecture: `n_blocks=8`, `couplings_per_block=2`, `hidden_dim=128`, `hidden_depth=5`
  - `hidden_depth=5` → MLP shape `[60, 128, 128, 128, 60]` (3 hidden layers)
  - `dt = 2 / (n_blocks × 2 × block_depth)` (flonaco rescaling convention)
- **Sigmoid output layer**: after the coupling layers, `forward` applies `sigmoid` to the first 51 dims (bonds/angles/torsions), ensuring flow samples always land in `(0, 1)`. `backward` applies the inverse `logit` before the coupling layers. Both account for the Jacobian log-determinant: `log|det J_sigmoid| = Σ log(σᵢ·(1−σᵢ))`. Without this, flow samples can fall outside `[0,1]`, the coordinate transform produces distorted geometries, and energies reach 10¹⁰–10¹⁵ kBT, making MH acceptance permanently zero.
- **No pre-training.** The flow starts from random weights and is trained only on chain samples.

### 3. `ICEnergy` — energy wrapper

Wraps the bgflow `coordinate_transform` + `target_energy` with a clean IC-space API:

- `.U(z)` → `(batch,)` energies in kBT via IC→Cartesian→OpenMM  
- `.grad_and_U(z)` → `(grad, U)` — single autograd pass; grad flows through the differentiable `coordinate_transform` and bgflow's `_BridgeEnergyWrapper` custom autograd function that injects OpenMM forces

### 4. `run_MALA_IC` — MALA with MH correction

Key details:
- **Step size**: `dt = 3e-6` gives ~43% acceptance empirically (IC gradient norms are ~500–2000 due to stiff MM bonds; much larger dt than Euclidean MALA)
- **Gradient clipping**: `MALA_GRAD_CLIP = 200.0` element-wise, applied **identically** in both proposal and MH ratio (so detailed balance is preserved for the clipped kernel)
- **IC clamping**: bonds/angles/torsions (first 51 dims) clamped to `(1e-6, 1-1e-6)` after every proposal step

MH ratio:
```
log α = -U(z') - ‖z - z' + dt·f(z')‖²/(4dt)
        + U(z)  + ‖z'- z  + dt·f(z )‖²/(4dt)
```
where `f(z) = clip(∇U(z), ±MALA_GRAD_CLIP)`.

### 5. `run_metropolis_IC` — flow-based independent MH

```
log α = -U(z') + nll_q(z') + U(z) - nll_q(z)
```

This is the independent MH kernel with proposal `q`. Acceptance rate starts near 0 and rises as the flow learns.

### 6. `train_adaptive` — main adaptive training loop

Each iteration:
1. **Sample phase**: `n_steps` MALA steps, with a flow MH jump every `lag` steps (inline, not via `run_metropolis_IC`, to enable U caching)
2. **U caching**: `grad_and_U` returns both gradient and per-sample U(z) in one pass; U values are updated in-place after each MALA accept/reject — the training phase reads `U_cache` with no extra OpenMM calls (~33% reduction in calls)
3. **Train phase**: `loss = mean(nll_q(z_batch) - U_batch)` — forward KL; gradient flows only through `nll_q`; `U_batch` is treated as a constant (detached)

**Logged metrics every iteration** (printed as a table):

| metric | meaning | expected behavior |
|--------|---------|-------------------|
| `loss` | forward KL estimate | decreases |
| `mala_acc` | MALA acceptance rate | stays ~0.30–0.50 |
| `mh_acc` | flow MH acceptance rate | starts ~0, should rise to >0.10 as flow learns |
| `nll_test` | flow NLL on held-out MD IC samples | decreases (independent of training data) |
| `U_flow` | mean energy of 50 fresh flow samples | should approach `U_chain` |
| `U_chain` | mean energy of chain positions | reference; relatively stable |

`eval_z_md` is a held-out set of 1000 MD IC samples used **only** for `nll_test`; it is never used for training.

### 7. `run_mala_only` — MALA baseline

Identical chain count, `dt`, and step budget as `train_adaptive`. No flow involved. Used for Ramachandran comparison.

---

## Key constants

```python
MALA_GRAD_CLIP = 200.0   # per-element gradient clip
_N_BOUNDED     = 51      # dims 0:51 are bonds/angles/torsions, must stay in (0,1)
_IC_EPS        = 1e-6    # clamping epsilon
```

Default hyperparameters (in `__main__`):
```python
N_ITER=500, N_STEPS=20, LAG=5, DT=3e-6, LR=1e-3
N_BLOCKS=8, HIDDEN=128, H_DEPTH=5, N_CHAINS=100
```

---

## External library pointers

| What you need | Where to look |
|---------------|---------------|
| RealNVP, MLP reference | `external/flonaco/flonaco/real_nvp_mlp.py`, `models.py` |
| Reference MALA (`run_MALA`) | `external/flonaco/flonaco/sampling.py:38–72` |
| Reference flow MH (`run_metropolis`) | `external/flonaco/flonaco/sampling.py:151–167` |
| Reference training loop | `external/flonaco/flonaco/training.py:212–275` (`mhmalangevin` mode) |
| IC→Cartesian inverse | `external/bgflow/bgflow/nn/flow/crd_transform/ic.py:862` (`_inverse`) |
| OpenMM autograd hook | `external/bgflow/bgflow/nn/flow/crd_transform/ic.py` (`_BridgeEnergyWrapper`) |
| ESS computation | `external/flonaco/flonaco/sampling.py:351` (`compute_ESS`) |

---

## Common gotchas

**Wrong Python**: `python` / `python3` in the shell are the system interpreter without PyTorch. Always use `~/.venvs/310/bin/python`.

**IC direction**: `coordinate_transform.forward(..., inverse=True)` is the **IC → Cartesian** direction. The name is confusing because `forward` in bgflow's base class dispatches to `_inverse` when the flag is set.

**Step size**: `dt=3e-6` is correct for IC space. The IC gradient norms are ~500–2000 (not ~1) because of stiff harmonic bond terms in the MM force field. The step size is three orders of magnitude smaller than a typical Euclidean MALA.

**Gradient clipping and detailed balance**: The clipped gradient `f(z) = clip(∇U, ±200)` must be used in **both** the proposal step and the MH log-ratio. Using raw gradients in the ratio but clipped gradients in the proposal would break detailed balance.

**Flow pre-training forbidden**: The comparison between MALA-only and MALA+Flow is only fair if the flow starts from random weights and sees only chain samples — no pre-training on the 1M MD frames.

**OpenMM pool startup**: `n_workers=os.cpu_count()` spawns a multiprocessing pool on first energy evaluation. There is a ~10–30 s one-time startup delay before any output appears.
