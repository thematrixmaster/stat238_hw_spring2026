"""
Adaptive Monte Carlo with Normalizing Flows on Alanine Dipeptide
Implements Gabrie et al. (2022) and compares against MALA-only baseline.

All MCMC and flow components are implemented in plain PyTorch.
bgflow / bgmol are used only as utilities for the coordinate transform and
the OpenMM energy function.
"""

import sys
import os
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "bgflow"))
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "bgmol"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

# ── Device / dtype ─────────────────────────────────────────────────────────────

n_workers = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype  = torch.float32
ctx    = torch.zeros([], device=device, dtype=dtype)
print(f"Running on device: {device}, dtype: {dtype}")


# ── Load dataset and energy model ──────────────────────────────────────────────

from bgmol.datasets import Ala2TSF300
import bgflow as bg

root        = Path(__file__).parent.parent / "data"
is_present  = os.path.isfile(root / "Ala2TSF300.npy")
assert root.is_dir(), f"Data directory {root} does not exist."

dataset       = Ala2TSF300(root=str(root), download=(not is_present), read=True)
dim           = dataset.dim          # 66 (22 atoms × 3 Cartesian)
system        = dataset.system
coordinates   = dataset.coordinates  # (1_000_000, 22, 3)
temperature   = dataset.temperature
target_energy = dataset.get_energy_model(n_workers=n_workers)  # OpenMMEnergy

print(system)
print(f"Coordinates shape: {coordinates.shape}")


# ── Coordinate transform (Cartesian ↔ mixed IC space) ─────────────────────────

dim_cartesian = len(system.rigid_block) * 3 - 6   # 9  (whitened backbone)
dim_bonds     = len(system.z_matrix)               # 17
dim_angles    = dim_bonds                          # 17
dim_torsions  = dim_bonds                          # 17
dim_ics       = dim_bonds + dim_angles + dim_torsions + dim_cartesian  # 60

all_data = torch.from_numpy(coordinates).to(ctx).reshape(-1, dim)

coordinate_transform = bg.MixedCoordinateTransformation(
    data=all_data,
    z_matrix=system.z_matrix,
    fixed_atoms=system.rigid_block,
    keepdims=dim_cartesian,
    normalize_angles=True,
).to(ctx)

# Sanity check
with torch.no_grad():
    _b, _a, _t, _c, _d = coordinate_transform.forward(all_data[:3])
assert _b.shape == (3, 17) and _c.shape == (3, 9), "Unexpected IC shapes"
print(f"IC dimension: {dim_ics}  "
      f"(bonds={dim_bonds}, angles={dim_angles}, "
      f"torsions={dim_torsions}, whitened_cart={dim_cartesian})")


# ═══════════════════════════════════════════════════════════════════════════════
# Normalizing Flow: RealNVP in IC space
# Adapted from flonaco/flonaco/{models.py, real_nvp_mlp.py}
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Feed-forward network with ReLU activations."""

    def __init__(self, layerdims, activation=torch.relu, init_scale=None):
        super().__init__()
        self.activation = activation
        linears = [nn.Linear(layerdims[i], layerdims[i + 1])
                   for i in range(len(layerdims) - 1)]
        if init_scale is not None:
            for l, layer in enumerate(linears):
                nn.init.normal_(layer.weight, std=init_scale / math.sqrt(layerdims[l]))
                nn.init.zeros_(layer.bias)
        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        layers = list(self.linears)
        for layer in layers[:-1]:
            x = self.activation(layer(x))
        return layers[-1](x)


class ResidualAffineCoupling(nn.Module):
    """
    Affine coupling layer with dt rescaling (Dinh et al., flonaco convention).

    mask:  binary (1, dim) tensor — 1 marks dims that are *transformed*,
           0 marks dims used as *conditioning input*.
    dt:    scale factor on s and t networks (controls expressiveness per layer).
    """

    def __init__(self, s, t, mask, dt=1.0):
        super().__init__()
        self.scale_net = s
        self.trans_net = t
        self.register_buffer("mask", mask)
        self.dt = dt

    def forward(self, x, log_det_jac=None, inverse=False):
        if log_det_jac is None:
            log_det_jac = torch.zeros(x.shape[0], device=x.device)

        # Conditioning = complement of mask; transform = mask
        x_cond = x * (1.0 - self.mask)
        s = self.mask * torch.tanh(self.scale_net(x_cond)) * self.dt
        t = self.mask * self.trans_net(x_cond) * self.dt

        if inverse:
            # Normalizing direction: x_data → z_prior  (log|det| decreases)
            log_det_jac -= s.view(x.shape[0], -1).sum(-1)
            x = x * torch.exp(-s) - t
        else:
            # Generative direction: z_prior → x_data  (log|det| increases)
            log_det_jac += s.view(x.shape[0], -1).sum(-1)
            x = (x + t) * torch.exp(s)

        return x, log_det_jac


class FlowIC(nn.Module):
    """
    RealNVP normalizing flow operating in the 60-dim mixed IC space.

    Convention:
      forward(z_prior) → (z_IC, log_det)      generative direction
      backward(z_IC)   → (z_prior, log_det)   normalizing direction
      nll(z_IC)        → (batch,) NLL values   -log q(z_IC)
      sample(n)        → (n, dim_ics)          z_IC samples from q
    """

    def __init__(self, dim, n_blocks=8, block_depth=1,
                 hidden_dim=128, hidden_depth=5,
                 mask_type="half", init_weight_scale=None):
        super().__init__()
        self.dim            = dim
        self.n_blocks       = n_blocks
        self.block_depth    = block_depth
        self.couplings_per_block = 2

        # Build alternating binary mask
        mask = torch.ones(dim)
        if mask_type == "half":
            mask[: dim // 2] = 0.0
        elif mask_type == "inter":
            mask = (torch.arange(dim) % 2 == 0).float()
        else:
            raise ValueError("mask_type must be 'half' or 'inter'")
        self.register_buffer("mask", mask.view(1, dim))

        # dt rescaling factor (same formula as flonaco real_nvp_mlp.py:236-237)
        dt = 2.0 / (n_blocks * self.couplings_per_block * block_depth)

        # Build coupling layers: list of blocks, each block has couplings_per_block layers
        # hidden_depth=5 gives MLP [dim, h, h, h, dim]  (3 hidden layers)
        layer_dims = [dim] + [hidden_dim] * (hidden_depth - 2) + [dim]

        blocks = []
        for _ in range(n_blocks):
            block = nn.ModuleList()
            for k in range(self.couplings_per_block):
                m = (1.0 - self.mask) if k % 2 == 0 else self.mask
                s = MLP(layer_dims, init_scale=init_weight_scale)
                t = MLP(layer_dims, init_scale=init_weight_scale)
                block.append(ResidualAffineCoupling(s, t, m, dt=dt))
            blocks.append(block)
        self.coupling_layers = nn.ModuleList(blocks)

    def forward(self, x):
        """Generative direction: z_prior → z_IC."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        for block in self.coupling_layers:
            for _ in range(self.block_depth):
                for coupling in block:
                    x, log_det = coupling(x, log_det)

        # Sigmoid output layer for the 51 bounded IC dimensions (bonds/angles/torsions).
        # Maps ℝ → (0,1), ensuring samples are always in the valid IC range.
        # log|det J_sigmoid| = Σ log(σᵢ·(1−σᵢ))  [negative, since σ compresses]
        x_b = torch.sigmoid(x[:, :_N_BOUNDED])
        log_det += torch.log(x_b * (1.0 - x_b)).sum(-1)
        x = torch.cat([x_b, x[:, _N_BOUNDED:]], dim=-1)
        return x, log_det

    def backward(self, x):
        """Normalizing direction: z_IC → z_prior (for NLL computation)."""
        log_det = torch.zeros(x.shape[0], device=x.device)

        # Logit layer: inverse of the sigmoid in forward.
        # Maps (0,1) → ℝ for the 51 bounded dims before the coupling layers.
        # log|det J_logit| = Σ log(1/(xᵢ·(1−xᵢ)))  [positive, since logit expands]
        x_b = x[:, :_N_BOUNDED].clamp(_IC_EPS, 1.0 - _IC_EPS)
        log_det += -torch.log(x_b * (1.0 - x_b)).sum(-1)
        x = torch.cat([torch.logit(x_b), x[:, _N_BOUNDED:]], dim=-1)

        for block in reversed(list(self.coupling_layers)):
            for _ in range(self.block_depth):
                for coupling in reversed(list(block)):
                    x, log_det = coupling(x, log_det, inverse=True)
        return x, log_det

    def nll(self, x):
        """Returns -log q(x) per sample via change-of-variables."""
        z, log_det = self.backward(x)
        # log p_prior(z) = -0.5 * ||z||^2 - 0.5 * dim * log(2π)
        prior_ll = -0.5 * (z ** 2).sum(-1) - 0.5 * self.dim * math.log(2.0 * math.pi)
        return -(prior_ll + log_det)

    @torch.no_grad()
    def sample(self, n):
        """Sample z_IC ~ q by pushing prior samples through the generative direction."""
        z = torch.randn(n, self.dim, device=self.mask.device)
        return self.forward(z)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Energy wrapper in IC space
# ═══════════════════════════════════════════════════════════════════════════════

# Number of IC dimensions whose values must stay in (0, 1)
_N_BOUNDED    = dim_bonds + dim_angles + dim_torsions   # 51
_IC_EPS       = 1e-6
# Max per-element gradient magnitude for MALA proposals.
# Energies in IC space have steep walls; clipping prevents runaway proposals
# while preserving detailed balance (the clipped gradient is used consistently
# in both the proposal and the MH ratio).
MALA_GRAD_CLIP = 200.0


def _clamp_ics(z: torch.Tensor) -> torch.Tensor:
    """Clamp bonds/angles/torsions (first 51 dims) to (ε, 1-ε). Whitened Cartesian left free."""
    z_c = z.clone()
    z_c[:, :_N_BOUNDED] = z_c[:, :_N_BOUNDED].clamp(_IC_EPS, 1.0 - _IC_EPS)
    return z_c


class ICEnergy:
    """
    Provides U(z) and grad_U(z) for z ∈ ℝ^{dim_ics}.

    z is laid out as:  [bonds (17) | angles (17) | torsions (17) | z_fixed (9)]

    U is returned in kBT units (bgflow normalises by kT internally), so the
    Boltzmann weight is exp(-U) with β = 1.
    """

    def __init__(self, coordinate_transform, target_energy,
                 d_b=17, d_a=17, d_t=17):
        self.ct   = coordinate_transform
        self.te   = target_energy
        self.d_b  = d_b
        self.d_a  = d_a
        self.d_t  = d_t

    def _split(self, z):
        b, a, t = self.d_b, self.d_a, self.d_t
        return z[:, :b], z[:, b:b+a], z[:, b+a:b+a+t], z[:, b+a+t:]

    def U(self, z: torch.Tensor) -> torch.Tensor:
        """
        Potential energy in kBT.  Shape: (batch,)
        Reconstructs Cartesian x from IC vector z, then calls OpenMM.
        """
        bonds, angles, torsions, z_fixed = self._split(z)
        # coordinate_transform.forward(..., inverse=True) is the IC → Cartesian direction
        x, _ = self.ct.forward(bonds, angles, torsions, z_fixed, inverse=True)
        return self.te.energy(x).squeeze(-1)   # (batch,)

    def grad_U(self, z: torch.Tensor) -> torch.Tensor:
        """Gradient ∂U/∂z. Shape: (batch, dim_ics)"""
        return self.grad_and_U(z)[0]

    def grad_and_U(self, z: torch.Tensor):
        """
        Returns (grad, U_per_sample) in a single forward+backward pass.
        More efficient than calling grad_U and U separately.
        Shape: (batch, dim_ics), (batch,)
        """
        z_req = z.detach().requires_grad_(True)
        with torch.enable_grad():
            # bgflow's _BridgeEnergy._energy caches results by tensor hash.
            # On a cache hit it returns a detached torch.tensor() with no grad_fn,
            # which breaks torch.autograd.grad.  Resetting _last_batch forces a
            # fresh call to _BridgeEnergyWrapper.apply(), which properly registers
            # the computation in the autograd graph.
            self.te._last_batch = None
            u = self.U(z_req)                          # (batch,) — keep per-sample
            grad = torch.autograd.grad(u.sum(), z_req)[0]
        return grad.detach(), u.detach()


ic_energy = ICEnergy(coordinate_transform, target_energy,
                     d_b=dim_bonds, d_a=dim_angles, d_t=dim_torsions)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0 — Seed the MCMC chains from MD data
# ═══════════════════════════════════════════════════════════════════════════════

n_chains = 200
stride   = len(all_data) // n_chains
x_seed   = all_data[::stride][:n_chains]

with torch.no_grad():
    b0, a0, t0, c0, _ = coordinate_transform.forward(x_seed)
    z_init = torch.cat([b0, a0, t0, c0], dim=-1)   # (n_chains, 60)

assert z_init.shape == (n_chains, dim_ics)
print(f"\nSeeded {n_chains} chains; IC shape: {z_init.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# MALA in IC space
# Adapted from flonaco/flonaco/sampling.py:38-72  (run_MALA)
# ═══════════════════════════════════════════════════════════════════════════════

def run_MALA_IC(ic_energy: ICEnergy, z_init: torch.Tensor,
                n_steps: int, dt: float,
                grad_clip: float = MALA_GRAD_CLIP):
    """
    MALA with Metropolis-Hastings correction in IC space.

    Proposal:   z' = z - dt·f(z) + √(2dt)·ε,   ε ~ N(0,I)
    where f(z) = clip(∇U(z), -grad_clip, +grad_clip)  (element-wise)

    The clipped gradient f is used consistently in both the proposal and the
    MH ratio, so detailed balance is exactly preserved for this modified
    proposal kernel.

    log α  =  -U(z') + U(z)
              - ‖z - z' + dt·f(z')‖² / (4dt)
              + ‖z'- z  + dt·f(z )‖² / (4dt)

    Args:
        ic_energy: ICEnergy instance
        z_init:    (n_chains, dim_ics) starting positions
        n_steps:   number of MALA steps
        dt:        step size — tune for ~30-50% acceptance rate.
                   Typical IC gradient norms are ~500-2000 (after clipping);
                   dt ≈ 3e-6 gives ~43% acceptance empirically.
        grad_clip: per-element gradient clip (prevents runaway proposals at
                   high-energy configurations near IC boundaries)

    Returns:
        xs:   (n_steps, n_chains, dim_ics) trajectory
        accs: (n_steps, n_chains) bool acceptance mask
    """
    xs, accs = [], []
    z        = z_init.clone()
    sqrt_2dt = math.sqrt(2.0 * dt)

    for _ in range(n_steps):
        g      = ic_energy.grad_U(z).clamp(-grad_clip, grad_clip)
        z_prop = _clamp_ics(z - dt * g + sqrt_2dt * torch.randn_like(z))

        g_prop   = ic_energy.grad_U(z_prop).clamp(-grad_clip, grad_clip)
        U_z      = ic_energy.U(z)
        U_z_prop = ic_energy.U(z_prop)

        # Squared norms for the forward and reverse proposal kernels
        fwd = ((z_prop - z + dt * g)      ** 2).sum(-1)   # ‖z'- z + dt·f(z )‖²
        rev = ((z      - z_prop + dt * g_prop) ** 2).sum(-1)  # ‖z - z'+ dt·f(z')‖²

        log_alpha = -U_z_prop - rev / (4.0 * dt) + U_z + fwd / (4.0 * dt)
        acc       = torch.log(torch.rand(z.shape[0])) < log_alpha.clamp(max=0.0)

        z_new        = z_prop.clone()
        z_new[~acc]  = z[~acc]

        xs.append(z_new.clone())
        accs.append(acc)
        z = z_new

    return torch.stack(xs), torch.stack(accs)


# ═══════════════════════════════════════════════════════════════════════════════
# Flow-based independent Metropolis-Hastings
# Adapted from flonaco/flonaco/sampling.py:151-167  (run_metropolis)
# ═══════════════════════════════════════════════════════════════════════════════

def run_metropolis_IC(flow: FlowIC, ic_energy: ICEnergy,
                      z_init: torch.Tensor, n_steps: int):
    """
    Independent MH using the normalizing flow q as proposal.

    Proposal:   z' ~ q(z)
    log α  =  -U(z') + nll_q(z') + U(z) - nll_q(z)
            =  log[p(z') / q(z')] - log[p(z) / q(z)]

    Args:
        flow:      FlowIC instance (the current learned proposal)
        ic_energy: ICEnergy instance
        z_init:    (n_chains, dim_ics) current positions
        n_steps:   number of MH steps

    Returns:
        xs:   (n_steps, n_chains, dim_ics)
        accs: (n_steps, n_chains) bool acceptance mask
    """
    xs, accs = [], []
    z = z_init.clone()

    for _ in range(n_steps):
        z_prop = flow.sample(z.shape[0])

        with torch.no_grad():
            log_alpha = (-ic_energy.U(z_prop) + flow.nll(z_prop)
                         + ic_energy.U(z)     - flow.nll(z))

        acc = torch.log(torch.rand(z.shape[0])) < log_alpha.clamp(max=0.0)
        z_new       = z_prop.clone()
        z_new[~acc] = z[~acc]

        xs.append(z_new.clone())
        accs.append(acc)
        z = z_new

    return torch.stack(xs), torch.stack(accs)


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive training loop  (Gabrie et al. 2022)
# Adapted from flonaco/flonaco/training.py:212-275  (mhmalangevin mode)
# ═══════════════════════════════════════════════════════════════════════════════

def train_adaptive(flow: FlowIC, ic_energy: ICEnergy, z_init: torch.Tensor,
                   n_iter: int = 500, lr: float = 1e-3,
                   n_steps: int = 20, lag: int = 5,
                   dt: float = 3e-6, grad_clip: float = 1e4,
                   eval_z_md: torch.Tensor = None,
                   print_every: int = 1):
    """
    Adaptive MCMC (Gabrie et al. 2022) with dense diagnostic logging.

    Each iteration:
      1. Run n_steps interleaved MALA + flow MH steps (MH every `lag` MALA steps).
      2. Collect all n_chains × n_steps IC samples.
      3. Train the flow via forward-KL loss on those samples.

    U values are cached across MALA steps so the training phase requires no
    additional energy evaluations (saves ~33% of total OpenMM calls).

    Logged metrics every iteration
    ────────────────────────────────
    loss       Forward-KL estimate E[nll_q(z) - U(z)]. Measures how well
               q approximates the target; should decrease over iterations.
    mala_acc   MALA acceptance rate. Should stay ~0.30-0.50 throughout.
               A sudden drop means a chain wandered to high-energy regions.
    mh_acc     Flow MH acceptance rate. Starts near 0 with a random flow.
               Rising mh_acc is the primary signal that the flow is learning.
    nll_test   Flow NLL on a held-out set of MD IC samples (not used for
               training). Measures fit quality independently of the chains.
               Should decrease as the flow learns the IC distribution.
    U_flow     Mean energy of 50 fresh flow samples (kBT). Compares to
               U_chain: if similar, flow proposals are energetically realistic.
    U_chain    Mean energy of chain positions this iteration (kBT).

    Args:
        flow:        FlowIC (randomly initialized, no pre-training)
        ic_energy:   ICEnergy instance
        z_init:      (n_chains, dim_ics) starting positions
        n_iter:      number of training iterations
        lr:          Adam learning rate
        n_steps:     MALA steps per iteration (= samples collected per chain)
        lag:         flow MH attempt every `lag` MALA steps
        dt:          MALA step size (3e-6 gives ~43% acceptance empirically)
        grad_clip:   max gradient norm for flow parameter updates
        eval_z_md:   optional (n_eval, dim_ics) held-out MD IC samples for
                     nll_test metric. Does not affect training.
        print_every: print frequency (iterations)

    Returns:
        z_final: (n_chains, dim_ics) final chain positions
        history: dict with per-iteration lists for all logged metrics
    """
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    z         = z_init.clone()
    sqrt_2dt  = math.sqrt(2.0 * dt)

    # Initialize U cache — one evaluation at start, then updated each step.
    # This means the training phase never needs a separate U(z_batch) call.
    U_z = ic_energy.U(z)

    history = {k: [] for k in
               ("loss", "mala_acc", "mh_acc", "nll_test", "U_flow", "U_chain")}

    # Print header
    print(f"\n{'t':>5} | {'loss':>9} | {'MALA_acc':>8} | {'MH_acc':>6} | "
          f"{'nll_test':>8} | {'U_flow':>7} | {'U_chain':>7}")
    print("-" * 70)

    for t in range(n_iter):
        xs_iter     = []
        U_cache     = []    # U(z_new) for each accepted MALA position
        mala_accs_t = []
        mh_accs_t   = []

        # ── Sample phase ──────────────────────────────────────────────────────
        for step in range(n_steps):

            # ── Flow MH jump every `lag` MALA steps (inline for U caching) ──
            if step % lag == 0:
                with torch.no_grad():
                    z_prop_mh   = flow.sample(z.shape[0])
                    U_z_prop_mh = ic_energy.U(z_prop_mh)
                    log_a_mh    = (-U_z_prop_mh + flow.nll(z_prop_mh)
                                   + U_z         - flow.nll(z))
                    acc_mh      = torch.log(torch.rand(z.shape[0])) < log_a_mh.clamp(max=0.0)
                    z_new_mh          = z_prop_mh.clone()
                    z_new_mh[~acc_mh] = z[~acc_mh]
                    U_z = torch.where(acc_mh, U_z_prop_mh, U_z)   # update cache
                    z   = z_new_mh
                    mh_accs_t.append(acc_mh.float().mean().item())

            # ── MALA step: grad_and_U avoids a redundant forward pass ────────
            g,   U_z      = ic_energy.grad_and_U(z)
            g    = g.clamp(-MALA_GRAD_CLIP, MALA_GRAD_CLIP)
            z_prop        = _clamp_ics(z - dt * g + sqrt_2dt * torch.randn_like(z))
            g_prop, U_z_prop = ic_energy.grad_and_U(z_prop)
            g_prop = g_prop.clamp(-MALA_GRAD_CLIP, MALA_GRAD_CLIP)

            fwd = ((z_prop - z + dt * g)          ** 2).sum(-1)
            rev = ((z      - z_prop + dt * g_prop) ** 2).sum(-1)
            log_alpha = -U_z_prop - rev / (4.0 * dt) + U_z + fwd / (4.0 * dt)
            acc       = torch.log(torch.rand(z.shape[0])) < log_alpha.clamp(max=0.0)

            z_new       = z_prop.clone()
            z_new[~acc] = z[~acc]
            U_z = torch.where(acc, U_z_prop, U_z)   # update cache for next step

            mala_accs_t.append(acc.float().mean().item())
            xs_iter.append(z_new.clone())
            U_cache.append(U_z.clone())
            z = z_new

        # (n_steps * n_chains, dim_ics) — detached from computation graph
        z_batch = torch.stack(xs_iter).reshape(-1, z_init.shape[-1]).detach()
        # Cached energies — no extra OpenMM calls needed for training
        U_batch = torch.stack(U_cache).reshape(-1).detach()

        # ── Train phase ───────────────────────────────────────────────────────
        # Forward KL: E_{z~chains}[nll_q(z) - U(z)] ≈ KL(p ‖ q) + const.
        # Gradient flows only through nll_q; U_batch is treated as a constant.
        optimizer.zero_grad()
        loss = (flow.nll(z_batch) - U_batch).mean()
        loss.backward()
        clip_grad_norm_(flow.parameters(), max_norm=grad_clip)
        optimizer.step()

        # ── Evaluation metrics ────────────────────────────────────────────────
        mala_acc = float(np.mean(mala_accs_t))
        mh_acc   = float(np.mean(mh_accs_t)) if mh_accs_t else 0.0

        with torch.no_grad():
            # NLL on held-out MD samples: independent measure of flow quality.
            # Should decrease as the flow learns the IC distribution.
            nll_test = (flow.nll(eval_z_md).mean().item()
                        if eval_z_md is not None else float("nan"))

            # Energy of 50 fresh flow samples vs chain energy.
            # If the flow is learning, U_flow should approach U_chain over time.
            z_flow_eval = flow.sample(50)
            U_flow  = ic_energy.U(z_flow_eval).mean().item()
            U_chain = U_batch.mean().item()

        history["loss"].append(loss.item())
        history["mala_acc"].append(mala_acc)
        history["mh_acc"].append(mh_acc)
        history["nll_test"].append(nll_test)
        history["U_flow"].append(U_flow)
        history["U_chain"].append(U_chain)

        if t % print_every == 0:
            print(f"{t:5d} | {loss.item():9.1f} | {mala_acc:8.3f} | {mh_acc:6.3f} | "
                  f"{nll_test:8.1f} | {U_flow:7.1f} | {U_chain:7.1f}")

    return z, history


# ═══════════════════════════════════════════════════════════════════════════════
# Static pre-training experiment
# Train the flow on a fixed set of MD IC samples to measure how many samples
# and gradient steps are needed to generate low-energy conformations.
# This is a diagnostic — it uses ground-truth MD data and is NOT used in the
# adaptive comparison.
# ═══════════════════════════════════════════════════════════════════════════════

def pretrain_flow(flow: FlowIC, z_md: torch.Tensor,
                  n_steps: int = 5000, lr: float = 1e-3,
                  batch_size: int = 1000, grad_clip: float = 1e4,
                  eval_every: int = 100, n_eval_samples: int = 50,
                  ic_energy: ICEnergy = None) -> dict:
    """
    Train the flow via forward KL on a fixed set of MD IC samples.

    Used as a diagnostic to answer: given enough data and gradient steps,
    can the RealNVP architecture generate low-energy conformations?

    The loss is E_{z~MD}[nll_q(z)], pure maximum likelihood — no U(z) term
    because the MD samples are already Boltzmann-distributed so maximizing
    their likelihood is equivalent to minimizing KL(p_target ‖ q).

    Args:
        flow:            FlowIC to train (modified in place)
        z_md:            (N, dim_ics) IC vectors from the MD trajectory
        n_steps:         number of gradient steps
        lr:              Adam learning rate
        batch_size:      mini-batch size drawn randomly from z_md each step
        grad_clip:       max gradient norm
        eval_every:      evaluate U_flow and NLL every this many steps
        n_eval_samples:  number of flow samples to draw when evaluating U_flow
        ic_energy:       ICEnergy instance for U_flow evaluation (optional;
                         skipped if None to avoid OpenMM startup cost)

    Returns:
        history dict with 'step', 'nll_train', 'U_flow' (if ic_energy given)
    """
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    history   = {"step": [], "nll_train": [], "U_flow": []}

    print(f"\n── Pre-training flow on {len(z_md)} MD IC samples ──")
    print(f"   {n_steps} steps, batch={batch_size}, lr={lr}")
    print(f"{'step':>6} | {'nll_train':>10} | {'U_flow':>14} | {'U_flow_min':>14} | {'U_flow_median':>14}")
    print("-" * 76)

    for step in range(1, n_steps + 1):
        idx   = torch.randint(len(z_md), (batch_size,))
        batch = z_md[idx]

        optimizer.zero_grad()
        # Pure MLE on MD samples: minimise -log q(z) averaged over the batch.
        # Equivalent to forward KL when training data ~ p_target.
        loss = flow.nll(batch).mean()
        loss.backward()
        clip_grad_norm_(flow.parameters(), max_norm=grad_clip)
        optimizer.step()

        if step % eval_every == 0:
            with torch.no_grad():
                nll_val = flow.nll(z_md[:2000]).mean().item()
            U_val = float("nan")
            if ic_energy is not None:
                with torch.no_grad():
                    z_samp = flow.sample(n_eval_samples)
                    U_val  = ic_energy.U(z_samp).mean().item()
                    U_val_min = ic_energy.U(z_md).min().item()
                    U_val_median = ic_energy.U(z_md).median().item()
            history["step"].append(step)
            history["nll_train"].append(nll_val)
            history["U_flow"].append(U_val)
            history["U_flow_min"].append(U_val_min)
            history["U_flow_median"].append(U_val_median)
            print(f"{step:6d} | {nll_val:10.2f} | {U_val:14.1f} | {U_val_min:14.1f} | {U_val_median:14.1f}")

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# MALA-only baseline (identical chain count and step budget, no flow)
# ═══════════════════════════════════════════════════════════════════════════════

def run_mala_only(ic_energy: ICEnergy, z_init: torch.Tensor,
                  n_iter: int = 500, n_steps: int = 20,
                  dt: float = 2e-4, print_every: int = 50):
    """
    Pure MALA with the same (n_chains, n_iter, n_steps, dt) as train_adaptive.

    Returns:
        all_xs:   (n_iter * n_steps, n_chains, dim_ics) full trajectory
        all_accs: (n_iter * n_steps, n_chains) bool acceptance mask
        history:  dict with 'mala_acc' per iteration
    """
    z         = z_init.clone()
    all_xs    = []
    all_accs  = []
    history   = {"mala_acc": []}

    for t in range(n_iter):
        xs, accs = run_MALA_IC(ic_energy, z, n_steps, dt)
        all_xs.append(xs)
        all_accs.append(accs)
        z = xs[-1]
        history["mala_acc"].append(accs.float().mean().item())
        if t % print_every == 0:
            print(f"t={t:4d}  MALA acc={history['mala_acc'][-1]:.3f}")

    return torch.cat(all_xs, dim=0), torch.cat(all_accs, dim=0), history


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def ic_to_cartesian(z_samples: torch.Tensor) -> torch.Tensor:
    """Convert (batch, dim_ics) IC vectors to (batch, 66) flat Cartesian coordinates."""
    b, a, t = dim_bonds, dim_angles, dim_torsions
    bonds    = z_samples[:, :b]
    angles   = z_samples[:, b:b+a]
    torsions = z_samples[:, b+a:b+a+t]
    z_fixed  = z_samples[:, b+a+t:]
    with torch.no_grad():
        x, _ = coordinate_transform.forward(bonds, angles, torsions, z_fixed, inverse=True)
    return x


def plot_ramachandran(ax, z_samples: torch.Tensor, title: str = ""):
    """Ramachandran plot for a batch of IC vectors."""
    import mdtraj as md
    from matplotlib.colors import LogNorm

    x    = ic_to_cartesian(z_samples).cpu().numpy()
    traj = md.Trajectory(xyz=x.reshape(-1, 22, 3), topology=system.mdtraj_topology)
    phi, psi = system.compute_phi_psi(traj)

    ax.hist2d(phi, psi, bins=50, norm=LogNorm())
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi, math.pi)
    ax.set_xlabel("φ")
    ax.set_ylabel("ψ")
    ax.set_title(title)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pickle
    import matplotlib.pyplot as plt

    # ── Hyperparameters ───────────────────────────────────────────────────────
    N_ITER      = 2000   # outer iterations
    N_STEPS     = 20     # MALA steps per iteration
    LAG         = 5      # flow MH every LAG MALA steps
    DT          = 3e-6   # MALA step size (~43% acceptance with MALA_GRAD_CLIP=200)
    LR          = 1e-3   # Adam lr for the flow
    N_BLOCKS    = 10     # RealNVP blocks (each has 2 coupling layers)
    HIDDEN      = 256    # MLP hidden dim
    H_DEPTH     = 5      # MLP depth: [dim, h, h, h, dim]  (3 hidden layers)
    N_EVAL      = 1000   # held-out MD samples for nll_test metric

    # ── Held-out MD IC samples for nll_test (evaluation only, not training) ──
    # These measure whether the flow is learning the correct IC distribution
    # independently of the chains it's being trained on.
    eval_idx  = torch.randperm(len(all_data))[:N_EVAL]
    with torch.no_grad():
        _eb, _ea, _et, _ec, _ = coordinate_transform.forward(all_data[eval_idx])
        eval_z_md = torch.cat([_eb, _ea, _et, _ec], dim=-1)
    print(f"Held-out eval set: {eval_z_md.shape}")

    # ── Sanity check ─────────────────────────────────────────────────────────
    print("\n── Sanity check: energy + gradient on 3 seed configs ──")
    z_test    = z_init[:3].clone()
    g_t, U_t  = ic_energy.grad_and_U(z_test)
    print(f"  U          = {U_t.tolist()}")
    print(f"  ‖∇U‖       = {g_t.norm(dim=-1).tolist()}")
    flow_tmp  = FlowIC(dim=dim_ics, n_blocks=2, hidden_dim=32)
    print(f"  nll (rand) = {flow_tmp.nll(z_test).tolist()}")
    del flow_tmp

    # ── Pre-training diagnostic ───────────────────────────────────────────────
    # Train on a fixed MD IC dataset to find out how many samples / gradient
    # steps are needed before U_flow reaches physical values.  Results inform
    # the adaptive training design but are NOT used in the final comparison.
    N_PRETRAIN_STEPS = 10000   # gradient steps
    N_PRETRAIN_MD    = 100000  # MD IC samples to use as the static dataset

    pretrain_idx = torch.randperm(len(all_data))[:N_PRETRAIN_MD]
    with torch.no_grad():
        _pb, _pa, _pt, _pc, _ = coordinate_transform.forward(all_data[pretrain_idx])
        z_pretrain = torch.cat([_pb, _pa, _pt, _pc], dim=-1)
    print(f"Pre-training dataset: {z_pretrain.shape}")

    flow_pretrain = FlowIC(dim=dim_ics, n_blocks=N_BLOCKS,
                           hidden_dim=HIDDEN, hidden_depth=H_DEPTH)
    hist_pretrain = pretrain_flow(
        flow_pretrain, z_pretrain,
        n_steps=N_PRETRAIN_STEPS, lr=LR, batch_size=1000,
        eval_every=100, n_eval_samples=50, ic_energy=ic_energy,
    )

    out_pretrain_pkl = Path(__file__).parent / "hist_pretrain.pkl"
    with open(out_pretrain_pkl, "wb") as f:
        import pickle; pickle.dump(hist_pretrain, f)
    print(f"Pre-train history saved to {out_pretrain_pkl}")

    # ── Run both methods from identical starting conditions ───────────────────
    torch.manual_seed(42)
    z0 = z_init.clone()

    # ── 1. Adaptive MALA + Flow ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MALA + Flow  ({N_ITER} iters × {N_STEPS} steps × {n_chains} chains, lag={LAG})")
    print(f"{'='*70}")
    print(f"  Flow:  {N_BLOCKS} blocks × 2 couplings, hidden=[{', '.join([str(HIDDEN)]*(H_DEPTH-2))}]")
    print(f"  n_workers (OpenMM): {n_workers}")
    print(f"\nWhat to watch:")
    print(f"  mh_acc   — starts ~0, should rise as the flow learns (target: >0.10)")
    print(f"  nll_test — NLL on held-out MD samples; should decrease (flow fitting IC distribution)")
    print(f"  U_flow   — energy of flow samples; should approach U_chain (flow proposals are realistic)")
    print(f"  loss     — forward KL; should decrease\n")

    flow = FlowIC(dim=dim_ics, n_blocks=N_BLOCKS,
                  hidden_dim=HIDDEN, hidden_depth=H_DEPTH)
    z_final_adap, hist_adap = train_adaptive(
        flow, ic_energy, z0.clone(),
        n_iter=N_ITER, n_steps=N_STEPS, lag=LAG, dt=DT, lr=LR,
        eval_z_md=eval_z_md, print_every=1,
    )

    # ── 2. MALA-only baseline ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MALA only  ({N_ITER} iters × {N_STEPS} steps × {n_chains} chains)")
    print(f"{'='*70}")
    xs_mala, accs_mala, hist_mala = run_mala_only(
        ic_energy, z0.clone(),
        n_iter=N_ITER, n_steps=N_STEPS, dt=DT, print_every=50,
    )

    # ── Save raw history to disk for offline analysis ─────────────────────────
    out_pkl = Path(__file__).parent / "history.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({"mala": hist_mala, "adaptive": hist_adap}, f)
    print(f"\nHistory saved to {out_pkl}")

    # ── Final summary ─────────────────────────────────────────────────────────
    w = 20   # window for trailing averages
    print(f"\n{'='*70}")
    print(f"Final summary  (trailing {w}-iter average)")
    print(f"{'='*70}")
    def tavg(lst): return float(np.mean(lst[-w:]))
    print(f"  MALA only   — MALA acc: {tavg(hist_mala['mala_acc']):.3f}")
    print(f"  Adaptive    — MALA acc: {tavg(hist_adap['mala_acc']):.3f} | "
          f"MH acc: {tavg(hist_adap['mh_acc']):.3f} | "
          f"nll_test: {tavg(hist_adap['nll_test']):.1f} | "
          f"U_flow: {tavg(hist_adap['U_flow']):.1f} | "
          f"U_chain: {tavg(hist_adap['U_chain']):.1f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    iters = range(N_ITER)

    # Acceptance rates
    axes[0, 0].plot(iters, hist_adap["mala_acc"], alpha=0.4, color="tab:blue", label="MALA acc")
    axes[0, 0].plot(iters, hist_adap["mh_acc"],   alpha=0.4, color="tab:orange", label="Flow MH acc")
    # Smoothed overlays
    def smooth(x, w=10): return np.convolve(x, np.ones(w)/w, mode="valid")
    axes[0, 0].plot(smooth(hist_adap["mala_acc"]), color="tab:blue",   lw=2)
    axes[0, 0].plot(smooth(hist_adap["mh_acc"]),   color="tab:orange", lw=2)
    axes[0, 0].axhline(0, color="k", lw=0.5, linestyle="--")
    axes[0, 0].set_xlabel("Iteration"); axes[0, 0].set_ylabel("Acceptance rate")
    axes[0, 0].legend(); axes[0, 0].set_title("Acceptance rates (adaptive)")

    # Forward KL loss
    axes[0, 1].plot(hist_adap["loss"], alpha=0.4, color="tab:green")
    axes[0, 1].plot(smooth(hist_adap["loss"]), color="tab:green", lw=2)
    axes[0, 1].set_xlabel("Iteration"); axes[0, 1].set_ylabel("Forward KL  [kBT]")
    axes[0, 1].set_title("Flow training loss")

    # nll_test: is the flow fitting the IC distribution?
    axes[0, 2].plot(hist_adap["nll_test"], alpha=0.4, color="tab:red")
    axes[0, 2].plot(smooth(hist_adap["nll_test"]), color="tab:red", lw=2)
    axes[0, 2].set_xlabel("Iteration"); axes[0, 2].set_ylabel("NLL on held-out MD ICs  [kBT]")
    axes[0, 2].set_title("Flow NLL on MD data  (↓ = flow learning)")

    # U_flow vs U_chain: are flow samples energetically realistic?
    axes[1, 0].plot(smooth(hist_adap["U_chain"]), label="Chain (MALA)", color="tab:blue", lw=2)
    axes[1, 0].plot(smooth(hist_adap["U_flow"]),  label="Flow samples",  color="tab:orange", lw=2)
    axes[1, 0].set_xlabel("Iteration"); axes[1, 0].set_ylabel("Mean energy  [kBT]")
    axes[1, 0].legend(); axes[1, 0].set_title("Energy: flow samples vs chain  (should converge)")

    # Ramachandran: MALA only (last 25% of trajectory)
    n_last       = N_STEPS * (N_ITER // 4)
    z_mala_final = xs_mala[-n_last:].reshape(-1, dim_ics)
    plot_ramachandran(axes[1, 1], z_mala_final,
                      title=f"MALA only  (last {len(z_mala_final)} samples)")

    # Ramachandran: adaptive (collect samples post-training, no flow update)
    xs_adap_eval, _ = run_MALA_IC(ic_energy, z_final_adap,
                                   n_steps=n_last // n_chains, dt=DT)
    z_adap_final    = xs_adap_eval.reshape(-1, dim_ics)
    plot_ramachandran(axes[1, 2], z_adap_final,
                      title=f"MALA + Flow  (last {len(z_adap_final)} samples)")

    plt.tight_layout()
    out_fig = Path(__file__).parent / "comparison.png"
    plt.savefig(out_fig, dpi=120)
    print(f"Figure saved to {out_fig}")
    plt.show()
