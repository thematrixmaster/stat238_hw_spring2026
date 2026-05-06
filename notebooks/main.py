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
from torch.nn.utils import clip_grad_norm_
from torch.distributions.multivariate_normal import MultivariateNormal

# ── Device / dtype ─────────────────────────────────────────────────────────────

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype  = torch.float32
ctx    = torch.zeros([], device=device, dtype=dtype)
print(f"Running on device: {device}, dtype: {dtype}")


# ── Load dataset and energy model ──────────────────────────────────────────────

from bgmol.datasets import Ala2TSF300
from bgflow.distribution.energy.openmm import OpenMMEnergy
import bgflow as bg

root        = Path(__file__).parent.parent / "data"
is_present  = os.path.isfile(root / "Ala2TSF300.npy")
assert root.is_dir(), f"Data directory {root} does not exist."

dataset       = Ala2TSF300(root=str(root), download=(not is_present), read=True)
dim           = dataset.dim          # 66 (22 atoms × 3 Cartesian)
system        = dataset.system
coordinates   = dataset.coordinates  # (1_000_000, 22, 3)
temperature   = dataset.temperature
target_energy = dataset.get_energy_model(n_workers=1)  # OpenMMEnergy

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
        return x, log_det

    def backward(self, x):
        """Normalizing direction: z_IC → z_prior (for NLL computation)."""
        log_det = torch.zeros(x.shape[0], device=x.device)
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
        """
        Gradient ∂U/∂z in IC space via autograd through the coordinate transform
        and OpenMM's custom backward (forces from _BridgeEnergyWrapper).
        Shape: (batch, dim_ics)
        """
        z_req = z.detach().requires_grad_(True)
        with torch.enable_grad():
            u = self.U(z_req).sum()
            grad = torch.autograd.grad(u, z_req)[0]
        return grad.detach()


ic_energy = ICEnergy(coordinate_transform, target_energy,
                     d_b=dim_bonds, d_a=dim_angles, d_t=dim_torsions)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0 — Seed the MCMC chains from MD data
# ═══════════════════════════════════════════════════════════════════════════════

n_chains = 100
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
                   dt: float = 2e-4, grad_clip: float = 1e4,
                   print_every: int = 50):
    """
    Each outer iteration:
      1. Run n_steps steps of interleaved MALA + flow Metropolis
         (flow MH attempted every `lag` MALA steps, matching flonaco mhmalangevin).
      2. Collect all n_chains × n_steps IC samples from this iteration.
      3. Train the flow on those samples via forward-KL loss.

    Args:
        flow:        FlowIC (randomly initialized, no pre-training)
        ic_energy:   ICEnergy instance
        z_init:      (n_chains, dim_ics) starting positions (same as MALA baseline)
        n_iter:      number of training iterations
        lr:          Adam learning rate
        n_steps:     MALA steps per iteration
        lag:         flow MH attempt frequency (every `lag` MALA steps)
        dt:          MALA step size
        grad_clip:   max gradient norm for flow parameter updates
        print_every: logging frequency (iterations)

    Returns:
        z_final: (n_chains, dim_ics) final chain positions
        history: dict with keys 'loss', 'mala_acc', 'mh_acc'
    """
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    z         = z_init.clone()
    sqrt_2dt  = math.sqrt(2.0 * dt)
    history   = {"loss": [], "mala_acc": [], "mh_acc": []}

    for t in range(n_iter):
        xs_iter       = []
        mala_accs_t   = []
        mh_accs_t     = []

        # ── Sample phase ──────────────────────────────────────────────────────
        for step in range(n_steps):

            # Flow Metropolis jump every `lag` MALA steps
            if step % lag == 0:
                z_mh, mh_acc = run_metropolis_IC(flow, ic_energy, z, n_steps=1)
                z = z_mh[-1]
                mh_accs_t.append(mh_acc[-1].float().mean().item())

            # MALA step (inline to avoid double energy evaluations)
            g        = ic_energy.grad_U(z).clamp(-MALA_GRAD_CLIP, MALA_GRAD_CLIP)
            z_prop   = _clamp_ics(z - dt * g + sqrt_2dt * torch.randn_like(z))
            g_prop   = ic_energy.grad_U(z_prop).clamp(-MALA_GRAD_CLIP, MALA_GRAD_CLIP)
            U_z      = ic_energy.U(z)
            U_z_prop = ic_energy.U(z_prop)

            fwd = ((z_prop - z + dt * g)          ** 2).sum(-1)
            rev = ((z      - z_prop + dt * g_prop) ** 2).sum(-1)
            log_alpha = -U_z_prop - rev / (4.0 * dt) + U_z + fwd / (4.0 * dt)
            acc       = torch.log(torch.rand(z.shape[0])) < log_alpha.clamp(max=0.0)
            z_new       = z_prop.clone()
            z_new[~acc] = z[~acc]

            mala_accs_t.append(acc.float().mean().item())
            xs_iter.append(z_new.clone())
            z = z_new

        # Batch of IC samples from this iteration: (n_steps * n_chains, dim_ics)
        z_batch = torch.stack(xs_iter).reshape(-1, z_init.shape[-1]).detach()

        # ── Train phase ───────────────────────────────────────────────────────
        # Forward-KL estimate:  E_{z~chains}[ nll_q(z) - U(z) ]
        #   = E_{z~chains}[ -log q(z) + U(z) ]   ≈  KL(p_target ‖ q_flow)  + const
        # The -U(z) term is a constant w.r.t. flow params (no gradient through it),
        # but makes the loss value equal to the approximate KL divergence.
        optimizer.zero_grad()
        loss = (flow.nll(z_batch) - ic_energy.U(z_batch).detach()).mean()
        loss.backward()
        clip_grad_norm_(flow.parameters(), max_norm=grad_clip)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        history["loss"].append(loss.item())
        history["mala_acc"].append(float(np.mean(mala_accs_t)))
        history["mh_acc"].append(float(np.mean(mh_accs_t)) if mh_accs_t else 0.0)

        if t % print_every == 0:
            print(f"t={t:4d}  loss={loss.item():9.2f}  "
                  f"MALA acc={history['mala_acc'][-1]:.3f}  "
                  f"MH acc={history['mh_acc'][-1]:.3f}")

    return z, history


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
    import matplotlib.pyplot as plt

    # ── Hyperparameters ───────────────────────────────────────────────────────
    N_ITER   = 200    # outer iterations for both methods
    N_STEPS  = 20     # MALA steps per iteration
    LAG      = 5      # flow MH every LAG MALA steps in adaptive run
    DT       = 3e-6   # MALA step size — gives ~43% acceptance with MALA_GRAD_CLIP=200
    LR       = 1e-3   # Adam lr for the flow
    N_BLOCKS = 8      # RealNVP blocks (each has 2 coupling layers)
    HIDDEN   = 128    # MLP hidden dim
    H_DEPTH  = 5      # hidden_depth → [dim, h, h, h, dim]  (3 hidden layers)

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\n── Sanity check: energy and gradient on 3 seed configurations ──")
    z_test = z_init[:3].clone()
    U_test = ic_energy.U(z_test)
    g_test = ic_energy.grad_U(z_test)
    print(f"  U(z_test) = {U_test.tolist()}")
    print(f"  ‖∇U‖ per chain = {g_test.norm(dim=-1).tolist()}")

    # ── Run both methods from identical starting conditions ───────────────────
    torch.manual_seed(42)
    z0 = z_init.clone()    # shared initial state

    print(f"\n── MALA only  ({N_ITER} iters × {N_STEPS} steps × {n_chains} chains) ──")
    xs_mala, accs_mala, hist_mala = run_mala_only(
        ic_energy, z0.clone(),
        n_iter=N_ITER, n_steps=N_STEPS, dt=DT,
    )

    print(f"\n── MALA + Flow (adaptive, {N_ITER} iters × {N_STEPS} steps × {n_chains} chains) ──")
    flow = FlowIC(dim=dim_ics, n_blocks=N_BLOCKS,
                  hidden_dim=HIDDEN, hidden_depth=H_DEPTH)
    z_final_adap, hist_adap = train_adaptive(
        flow, ic_energy, z0.clone(),
        n_iter=N_ITER, n_steps=N_STEPS, lag=LAG, dt=DT, lr=LR,
    )

    # ── Collect final samples for Ramachandran comparison ────────────────────
    # Use the last quarter of the MALA trajectory (post burn-in)
    n_last = N_STEPS * (N_ITER // 4)
    z_mala_final = xs_mala[-n_last:].reshape(-1, dim_ics)

    # Collect the same number of samples from the adaptive chains (run without training)
    print("\nCollecting adaptive samples for Ramachandran plot...")
    xs_adap_eval, _ = run_MALA_IC(ic_energy, z_final_adap, n_steps=n_last // n_chains, dt=DT)
    z_adap_final    = xs_adap_eval.reshape(-1, dim_ics)

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Acceptance rates
    axes[0, 0].plot(hist_mala["mala_acc"], label="MALA acc (baseline)")
    axes[0, 0].plot(hist_adap["mala_acc"], label="MALA acc (adaptive)", alpha=0.7)
    axes[0, 0].plot(hist_adap["mh_acc"],   label="Flow MH acc (adaptive)", linestyle="--")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Acceptance rate")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Acceptance rates")

    # Forward KL loss
    axes[0, 1].plot(hist_adap["loss"])
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Forward KL loss  [approx. kBT]")
    axes[0, 1].set_title("Flow training loss")

    # Ramachandran: MALA only
    plot_ramachandran(axes[1, 0], z_mala_final,
                      title=f"MALA only  (last {len(z_mala_final)} samples)")

    # Ramachandran: adaptive
    plot_ramachandran(axes[1, 1], z_adap_final,
                      title=f"MALA + Flow  (last {len(z_adap_final)} samples)")

    plt.tight_layout()
    out = Path(__file__).parent / "comparison.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")
    plt.show()
