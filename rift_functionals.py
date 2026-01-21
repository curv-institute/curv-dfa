"""
RIFT Functionals: Fisher + SCC correction modules for PySCF.

This module provides two orthogonal correction channels:
  1. Fisher + Harmonizer: XC-side term for geometry/bonding control
  2. SCC (Self-Coupling Cancellation): Hartree-side term for SIE correction

Usage:
    from rift_functionals import FisherModule, SCCModule, FisherCfg, SCCCfg

    mf = dft.UKS(mol)
    mf.xc = "GGA_X_PBE,GGA_C_PBE"

    fisher = FisherModule(FisherCfg(alpha=0.01))
    fisher.attach(mf)

    scc = SCCModule(SCCCfg(lam=0.1))
    scc.attach(mf)

    mf.kernel()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
from pyscf.dft import libxc
from pyscf.dft.numint import NumInt


# =============================================================================
# Constants
# =============================================================================

BASE_XC = "GGA_X_PBE,GGA_C_PBE"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass(frozen=True)
class FisherCfg:
    """Configuration for Fisher + Harmonizer correction."""
    alpha: float = 0.0
    harmonize: bool = True
    rho_core: float = 1.0
    core_power: float = 4.0
    rho_floor: float = 1e-7

    def as_dict(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'harmonize': self.harmonize,
            'rho_core': self.rho_core,
            'core_power': self.core_power,
            'rho_floor': self.rho_floor,
        }


@dataclass(frozen=True)
class SCCCfg:
    """
    Configuration for Self-Coupling Cancellation v2.

    SCC applies a scaled Hartree correction in one-electron-like regions:
        J_eff = J[ρ] * (1 - λ * ⟨w_eff⟩)

    where w_eff = w(z) × h(ρ):
        - w(z): iso-orbital gate (z = τ_W/τ, high in single-orbital regions)
        - h(ρ): density suppression (kills correction in many-electron densities)

    This two-stage gate distinguishes "one-electron" from "one-orbital".
    """
    lam: float = 0.09        # tuned for CO/NO + H₂⁺ balance
    w_power: int = 4
    z_clip: float = 1e-12
    rho_floor: float = 1e-12
    # Density suppression gate h(ρ) = 1 / (1 + (ρ/ρ_s)^p)
    rho_s: float = 0.03      # density threshold (a.u.); tuned to suppress in CO/NO
    h_power: int = 6         # suppression sharpness

    def as_dict(self) -> Dict[str, Any]:
        return {
            'lam': self.lam,
            'w_power': self.w_power,
            'z_clip': self.z_clip,
            'rho_floor': self.rho_floor,
            'rho_s': self.rho_s,
            'h_power': self.h_power,
        }


# =============================================================================
# Helpers
# =============================================================================

def _spin_to_int(spin) -> int:
    """Convert spin argument to integer."""
    try:
        a = np.asarray(spin).ravel()
        return int(a[0]) if a.size else 0
    except Exception:
        return int(spin)


def _add_gga_vxc(vxc, vrho_add, vsigma_add):
    """
    Add RIFT contributions to RKS vxc.

    vxc may be:
      - tuple/list: (vrho, vsigma) or (vrho, vsigma, ...) with extra channels
      - ndarray: shape (2, ngrids) or (>=2, ngrids)
    Returns same structure with first two channels incremented.
    """
    if isinstance(vxc, (tuple, list)):
        if len(vxc) < 2:
            raise RuntimeError(f"Unexpected vxc tuple length: {len(vxc)}")
        vxc0 = np.asarray(vxc[0]) + vrho_add
        vxc1 = np.asarray(vxc[1]) + vsigma_add
        rest = list(vxc[2:])
        return (vxc0, vxc1, *rest)
    v = np.asarray(vxc)
    if v.ndim == 1:
        return v + vrho_add
    if v.shape[0] < 2:
        raise RuntimeError(f"Unexpected vxc array shape: {v.shape}")
    v = v.copy()
    v[0] += vrho_add
    v[1] += vsigma_add
    return v


def _add_uks_vxc(vxc, vrho_add, vsigma_add):
    """
    Add RIFT contributions to UKS vxc.

    For UKS GGA, libxc returns vxc as list:
      vxc[0]: vrho (ngrids, 2) for [alpha, beta]
      vxc[1]: vsigma (ngrids, 3) for [σ_αα, σ_αβ, σ_ββ]

    Since RIFT uses total density σ_tot = σ_αα + 2*σ_αβ + σ_ββ:
      ∂f/∂σ_αα = vsigma_add
      ∂f/∂σ_αβ = 2 * vsigma_add
      ∂f/∂σ_ββ = vsigma_add
    """
    if not isinstance(vxc, (tuple, list)) or len(vxc) < 2:
        raise RuntimeError(f"Unexpected UKS vxc structure: {type(vxc)}")

    vrho = np.asarray(vxc[0]).copy()
    vsigma = np.asarray(vxc[1]).copy()

    # vrho: (ngrids, 2) - add to both spin channels
    if vrho.ndim == 2:
        vrho[:, 0] += vrho_add
        vrho[:, 1] += vrho_add
    else:
        vrho += vrho_add

    # vsigma: (ngrids, 3) - [σ_αα, σ_αβ, σ_ββ]
    vsigma_arr = np.asarray(vsigma_add)
    if np.any(vsigma_arr != 0.0):
        if vsigma.ndim == 2:
            vsigma[:, 0] += vsigma_arr        # ∂/∂σ_αα
            vsigma[:, 1] += 2.0 * vsigma_arr  # ∂/∂σ_αβ
            vsigma[:, 2] += vsigma_arr        # ∂/∂σ_ββ
        else:
            vsigma += vsigma_arr

    rest = list(vxc[2:]) if len(vxc) > 2 else []
    return (vrho, vsigma, *rest)


# =============================================================================
# Fisher + Harmonizer XC Functional
# =============================================================================

def make_rift_xc(
    *,
    base_xc: str = BASE_XC,
    alpha: float = 0.0,
    beta: float = 0.0,
    rho0: float = 1.0,
    rho_floor: float = 1e-7,
    harmonize: bool = True,
    rho_core: float = 1.0,
    core_power: float = 4.0,
):
    """
    Create RIFT-corrected XC functional callable.

    E_xc = E_xc_base
         + alpha ∫ f(s) g(ρ) |∇ρ|²/ρ dr   (Fisher information term)
         + beta  ∫ ρ ln(ρ/rho0) dr          (entropy term)

    Gates (when enabled):
      f(s) = s² / (1 + s²)           (reduced-gradient gate, harmonize=True)
      g(ρ) = 1 / (1 + (ρ/ρ_c)^p)     (core-suppression gate, rho_core > 0)

    Parameters:
        base_xc: Base XC functional (libxc string)
        alpha: Fisher term coefficient
        beta: Entropy term coefficient
        rho0: Reference density for entropy
        rho_floor: Density floor for numerical stability
        harmonize: Enable reduced-gradient gate f(s)
        rho_core: Core suppression density scale (0 = disabled)
        core_power: Power for core gate

    Returns:
        XC callable for use with mf.define_xc_()
    """
    if rho0 <= 0:
        raise ValueError("rho0 must be > 0")
    if rho_floor <= 0:
        raise ValueError("rho_floor must be > 0")

    def xc_callable(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        spin_int = _spin_to_int(spin)
        if deriv != 1:
            raise NotImplementedError("RIFT XC callable supports deriv=1 only.")

        rho_arr = np.asarray(rho, dtype=float)

        # === UKS (spin-polarized) ===
        if spin_int != 0:
            if rho_arr.ndim == 2:
                ngrids = rho_arr.shape[1]
                gga_rho = np.zeros((2, 4, ngrids))
                gga_rho[:, 0, :] = rho_arr
                has_grad = False
            elif rho_arr.ndim == 3:
                gga_rho = rho_arr
                has_grad = gga_rho.shape[1] >= 4
            else:
                raise RuntimeError(f"Unexpected UKS rho shape: {rho_arr.shape}")

            exc, vxc, _, _ = libxc.eval_xc(base_xc, gga_rho, spin=spin_int, deriv=1)

            rho_a, rho_b = gga_rho[0, 0], gga_rho[1, 0]
            rho_tot = rho_a + rho_b
            rf = np.maximum(rho_tot, rho_floor)

            if alpha != 0.0 and has_grad:
                grad_a = gga_rho[0, 1:4]
                grad_b = gga_rho[1, 1:4]
                grad_tot = grad_a + grad_b
                sigma_tot = np.einsum("ig,ig->g", grad_tot, grad_tot)

                inv = 1.0 / rf
                inv2 = inv * inv

                f_s = 1.0
                g_rho = 1.0
                df_term = 0.0
                dg_term = 0.0

                if harmonize:
                    c_s = 2.0 * (3.0 * np.pi**2) ** (1.0/3.0)
                    grad_norm = np.sqrt(np.maximum(sigma_tot, 1e-30))
                    s = grad_norm / (c_s * rf ** (4.0/3.0))
                    s2 = s * s
                    f_s = s2 / (1.0 + s2)
                    df_term = (8.0/3.0) * s2 / (1.0 + s2) ** 2

                if rho_core > 0:
                    u = (rf / rho_core) ** core_power
                    g_rho = 1.0 / (1.0 + u)
                    dg_term = core_power * u * g_rho

                gate = f_s * g_rho
                exc = exc + alpha * gate * (sigma_tot * inv2)

                vrho_add = -alpha * (sigma_tot * inv2) * (gate + g_rho * df_term + f_s * dg_term * g_rho)

                if harmonize:
                    s_df_half = s2 / (1.0 + s2) ** 2
                    vsigma_add = alpha * inv * g_rho * (f_s + s_df_half)
                else:
                    vsigma_add = alpha * inv * g_rho

                vxc = _add_uks_vxc(vxc, vrho_add, vsigma_add)

            if beta != 0.0:
                ln = np.log(rf / rho0)
                exc = exc + beta * ln
                vrho_add = beta * (ln + 1.0)
                vxc = _add_uks_vxc(vxc, vrho_add, 0.0)

            return exc, vxc, None, None

        # === RKS (spin-unpolarized) ===
        if rho_arr.ndim == 0:
            rho_arr = rho_arr.reshape(1, 1)

        if rho_arr.ndim == 1:
            rho_g = rho_arr
            gga_rho = np.vstack([rho_g,
                                 np.zeros_like(rho_g),
                                 np.zeros_like(rho_g),
                                 np.zeros_like(rho_g)])
            has_grad = False
        elif rho_arr.ndim == 2 and rho_arr.shape[0] < 4:
            rho_g = rho_arr[0]
            gga_rho = np.vstack([rho_g,
                                 np.zeros_like(rho_g),
                                 np.zeros_like(rho_g),
                                 np.zeros_like(rho_g)])
            has_grad = False
        else:
            gga_rho = rho_arr
            rho_g = gga_rho[0]
            has_grad = True

        exc, vxc, _, _ = libxc.eval_xc(base_xc, gga_rho, spin=spin_int, deriv=1)

        grad = gga_rho[1:4]
        sigma = np.einsum("ig,ig->g", grad, grad)

        rf = np.maximum(rho_g, rho_floor)

        if alpha != 0.0 and has_grad:
            inv = 1.0 / rf
            inv2 = inv * inv

            f_s = 1.0
            g_rho = 1.0
            df_term = 0.0
            dg_term = 0.0

            if harmonize:
                c_s = 2.0 * (3.0 * np.pi**2) ** (1.0/3.0)
                grad_norm = np.sqrt(np.maximum(sigma, 1e-30))
                s = grad_norm / (c_s * rf ** (4.0/3.0))
                s2 = s * s
                f_s = s2 / (1.0 + s2)
                df_term = (8.0/3.0) * s2 / (1.0 + s2) ** 2

            if rho_core > 0:
                u = (rf / rho_core) ** core_power
                g_rho = 1.0 / (1.0 + u)
                dg_term = core_power * u * g_rho

            gate = f_s * g_rho
            exc = exc + alpha * gate * (sigma * inv2)

            vrho_add = -alpha * (sigma * inv2) * (gate + g_rho * df_term + f_s * dg_term * g_rho)

            if harmonize:
                s_df_half = s2 / (1.0 + s2) ** 2
                vsigma_add = alpha * inv * g_rho * (f_s + s_df_half)
            else:
                vsigma_add = alpha * inv * g_rho

            vxc = _add_gga_vxc(vxc, vrho_add, vsigma_add)

        if beta != 0.0:
            ln = np.log(rf / rho0)
            exc = exc + beta * ln
            vrho_add = beta * (ln + 1.0)
            vxc = _add_gga_vxc(vxc, vrho_add, 0.0)

        return exc, vxc, None, None

    return xc_callable


# =============================================================================
# SCC (Self-Coupling Cancellation)
# =============================================================================

def attach_scc_hartree(
    mf,
    *,
    lam: float = 0.1,
    w_power: int = 4,
    z_clip: float = 1e-12,
    rho_floor: float = 1e-12,
    rho_s: float = 0.1,
    h_power: int = 6,
):
    """
    Attach SCC v2 (self-coupling cancellation) to a PySCF UKS/RKS object.

    SCC modifies the Hartree potential via get_veff override:
        J_eff = J[ρ] * (1 - λ * ⟨w_eff⟩)

    where w_eff = w(z) × h(ρ) is a two-stage gate:
        - w(z): iso-orbital gate, z = τ_W/τ
        - h(ρ): density suppression, h = 1/(1 + (ρ/ρ_s)^p)

    This ensures SCC is only active in low-density, one-electron-like regions
    (e.g., H₂⁺ mid-bond) and NOT in high-density many-electron regions
    (e.g., CO lone pairs).

    Parameters:
        mf: PySCF RKS/UKS object
        lam: Strength parameter; ~0.1-0.5 recommended
        w_power: Gate sharpness for w(z) (default 4)
        z_clip: Floor for τ to avoid division by zero
        rho_floor: Floor for ρ in τ_W calculation
        rho_s: Density threshold for suppression (a.u.); ~0.05-0.2
        h_power: Suppression sharpness for h(ρ) (default 6)

    Returns:
        Modified mf object
    """
    if not (0.0 <= lam <= 2.0):
        raise ValueError("lam should be in [0, 2.0] for stability")
    if w_power < 1:
        raise ValueError("w_power must be >= 1")
    if rho_s <= 0:
        raise ValueError("rho_s must be > 0")

    ni = mf._numint if hasattr(mf, "_numint") else NumInt()
    grids = mf.grids

    get_veff_orig = mf.get_veff

    def gate_from_z(z):
        """Smooth gate w(z) ∈ [0,1]."""
        z = np.clip(z, 0.0, 1.0)
        zn = z ** w_power
        on = (1.0 - z) ** w_power
        return zn / (zn + on + 1e-30)

    def get_veff_scc(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None:
            mol = mf.mol
        if dm is None:
            dm = mf.make_rdm1()

        veff = get_veff_orig(mol, dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi)

        dm_arr = np.asarray(dm)
        is_uks = dm_arr.ndim == 3 and dm_arr.shape[0] == 2

        # Handle RKS vs UKS
        if dm_arr.ndim == 2:
            if not hasattr(mf, 'mo_coeff') or mf.mo_coeff is None:
                return veff
            # RKS path
            is_uks = False
        elif not is_uks:
            return veff

        if grids.coords is None or grids.weights is None:
            grids.build(with_non0tab=True)

        ao = ni.eval_ao(mol, grids.coords, deriv=1)
        ao0, aox, aoy, aoz = ao[0], ao[1], ao[2], ao[3]

        if is_uks:
            rho_a_full = ni.eval_rho(mol, ao, dm_arr[0], xctype="GGA", hermi=hermi)
            rho_b_full = ni.eval_rho(mol, ao, dm_arr[1], xctype="GGA", hermi=hermi)
            rho_tot = rho_a_full[0] + rho_b_full[0]
            grad_tot = rho_a_full[1:4] + rho_b_full[1:4]
            dm_tot = dm_arr[0] + dm_arr[1]
        else:
            rho_full = ni.eval_rho(mol, ao, dm_arr, xctype="GGA", hermi=hermi)
            rho_tot = rho_full[0]
            grad_tot = rho_full[1:4]
            dm_tot = dm_arr

        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        def tau_from_spin(coeff, occ):
            occ_mask = occ > 1e-12
            if not np.any(occ_mask):
                return np.zeros(len(grids.coords))
            Cocc = coeff[:, occ_mask]
            occv = occ[occ_mask]
            gx = aox @ Cocc
            gy = aoy @ Cocc
            gz = aoz @ Cocc
            g2 = (gx * gx + gy * gy + gz * gz) * occv
            return 0.5 * np.sum(g2, axis=1)

        if is_uks:
            tau_tot = tau_from_spin(mo_coeff[0], mo_occ[0]) + tau_from_spin(mo_coeff[1], mo_occ[1])
        else:
            tau_tot = tau_from_spin(mo_coeff, mo_occ)

        sigma_tot = np.einsum("ig,ig->g", grad_tot, grad_tot)
        rho_eff = np.maximum(rho_tot, rho_floor)
        tauW = sigma_tot / (8.0 * rho_eff)

        tau_eff = np.maximum(tau_tot, z_clip)
        z = np.clip(tauW / tau_eff, 0.0, 1.0)
        w = gate_from_z(z)

        # Density suppression gate: h(ρ) = 1 / (1 + (ρ/ρ_s)^p)
        # Kills SCC in high-density (many-electron) regions
        h = 1.0 / (1.0 + (rho_eff / rho_s) ** h_power)

        # Effective gate: w_eff = w(z) × h(ρ)
        w_eff = w * h

        J = mf.get_j(mol, dm_tot)

        wt = grids.weights
        rho_weighted = rho_tot * wt
        w_eff_avg = np.sum(w_eff * rho_weighted) / np.maximum(np.sum(rho_weighted), 1e-30)

        scale_factor = lam * w_eff_avg
        delta_J = -scale_factor * J

        veff_arr = np.asarray(veff)
        if veff_arr.ndim == 3:
            veff_new = veff_arr.copy()
            veff_new[0] = veff_new[0] + delta_J
            veff_new[1] = veff_new[1] + delta_J
        else:
            veff_new = veff_arr + delta_J

        # Preserve special attributes
        if hasattr(veff, 'ecoul'):
            veff_new = np.asarray(veff_new).view(type(veff))
            veff_new.ecoul = veff.ecoul * (1.0 - scale_factor)
        if hasattr(veff, 'exc'):
            veff_new.exc = veff.exc
        if hasattr(veff, 'vj'):
            veff_new.vj = veff.vj
        if hasattr(veff, 'vk'):
            veff_new.vk = veff.vk

        return veff_new

    mf.get_veff = get_veff_scc
    return mf


# =============================================================================
# Module Classes (Uniform API)
# =============================================================================

class FisherModule:
    """Fisher + Harmonizer correction module."""

    def __init__(self, cfg: Optional[FisherCfg] = None):
        self.cfg = cfg or FisherCfg()

    @property
    def name(self) -> str:
        return f"Fisher(α={self.cfg.alpha}, harm={self.cfg.harmonize})"

    def attach(self, mf, base_xc: str = BASE_XC):
        """Attach Fisher correction to mf via define_xc_."""
        mf.define_xc_(
            make_rift_xc(
                base_xc=base_xc,
                alpha=self.cfg.alpha,
                harmonize=self.cfg.harmonize,
                rho_core=self.cfg.rho_core,
                core_power=self.cfg.core_power,
                rho_floor=self.cfg.rho_floor,
            ),
            xctype="GGA"
        )
        return mf


class SCCModule:
    """Self-Coupling Cancellation module."""

    def __init__(self, cfg: Optional[SCCCfg] = None):
        self.cfg = cfg or SCCCfg()

    @property
    def name(self) -> str:
        return f"SCC(λ={self.cfg.lam})"

    def attach(self, mf):
        """Attach SCC correction to mf via get_veff override."""
        attach_scc_hartree(
            mf,
            lam=self.cfg.lam,
            w_power=self.cfg.w_power,
            z_clip=self.cfg.z_clip,
            rho_floor=self.cfg.rho_floor,
            rho_s=self.cfg.rho_s,
            h_power=self.cfg.h_power,
        )
        return mf


# =============================================================================
# Diagnostic Primitives
# =============================================================================

def compute_rift_energy(mf, alpha, beta=0.0, rho0=1.0, rho_floor=1e-7,
                        harmonize=True, rho_core=1.0, core_power=4.0):
    """Compute RIFT correction energy components (post-SCF diagnostic)."""
    mol = mf.mol
    dm = mf.make_rdm1()
    grids = mf.grids

    ao = mf._numint.eval_ao(mol, grids.coords, deriv=1)
    rho_full = mf._numint.eval_rho(mol, ao, dm, xctype='GGA')
    rho_g = rho_full[0]
    grad = rho_full[1:4]
    sigma = np.einsum("ig,ig->g", grad, grad)
    weights = grids.weights

    rf = np.maximum(rho_g, rho_floor)

    e_fisher = 0.0
    e_entropy = 0.0

    if alpha != 0.0:
        f_s = np.ones_like(rf)
        g_rho = np.ones_like(rf)

        if harmonize:
            c_s = 2.0 * (3.0 * np.pi**2) ** (1.0/3.0)
            grad_norm = np.sqrt(np.maximum(sigma, 1e-30))
            s = grad_norm / (c_s * rf ** (4.0/3.0))
            s2 = s * s
            f_s = s2 / (1.0 + s2)

        if rho_core > 0:
            u = (rf / rho_core) ** core_power
            g_rho = 1.0 / (1.0 + u)

        gate = f_s * g_rho
        e_fisher = alpha * np.sum(weights * gate * sigma / rf)

    if beta != 0.0:
        ln = np.log(rf / rho0)
        e_entropy = beta * np.sum(weights * rho_g * ln)

    return e_fisher, e_entropy


def compute_radial_fisher_decomposition(mf, alpha, atom_idx=0, rho_floor=1e-7,
                                         harmonize=True, rho_core=1.0, core_power=4.0):
    """Decompose Fisher energy by radial shells around a specific atom."""
    mol = mf.mol
    dm = mf.make_rdm1()
    grids = mf.grids

    ao = mf._numint.eval_ao(mol, grids.coords, deriv=1)
    rho_full = mf._numint.eval_rho(mol, ao, dm, xctype='GGA')
    rho_g = rho_full[0]
    grad = rho_full[1:4]
    sigma = np.einsum("ig,ig->g", grad, grad)
    weights = grids.weights

    rf = np.maximum(rho_g, rho_floor)

    f_s = np.ones_like(rf)
    g_rho = np.ones_like(rf)

    if harmonize:
        c_s = 2.0 * (3.0 * np.pi**2) ** (1.0/3.0)
        grad_norm = np.sqrt(np.maximum(sigma, 1e-30))
        s = grad_norm / (c_s * rf ** (4.0/3.0))
        s2 = s * s
        f_s = s2 / (1.0 + s2)

    if rho_core > 0:
        u = (rf / rho_core) ** core_power
        g_rho = 1.0 / (1.0 + u)

    gate = f_s * g_rho
    fisher_density = alpha * weights * gate * sigma / rf

    atom_coord = mol.atom_coord(atom_idx)
    r = np.linalg.norm(grids.coords - atom_coord, axis=1)

    shells = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf]
    shell_energies = []

    for i in range(len(shells) - 1):
        mask = (r >= shells[i]) & (r < shells[i+1])
        e_shell = np.sum(fisher_density[mask])
        shell_energies.append((shells[i], shells[i+1], e_shell))

    return shell_energies, np.sum(fisher_density)


def get_orbital_energies(mf) -> Tuple[float, float, float]:
    """Extract HOMO, LUMO, and gap from converged calculation."""
    occ = mf.mo_occ
    mo_e = mf.mo_energy
    homo = mo_e[occ > 0].max()
    lumo = mo_e[occ == 0].min() if np.any(occ == 0) else np.nan
    gap = lumo - homo
    return homo, lumo, gap


def compute_density_diagnostics(mf_ref, mf_test, rho_floor=1e-7) -> Dict[str, Any]:
    """
    Compute density difference diagnostics between two converged calculations.

    Returns dict with:
      - L1: ∫|Δρ|dr (total redistribution)
      - L2: √(∫(Δρ)²dr) (sensitivity)
      - weighted: ∫(Δρ)²/(ρ_ref + rho_floor) dr
      - nel_ref, nel_test: electron counts
      - max_delta: max|Δρ|
      - max_delta_r: location of max|Δρ|
    """
    mol = mf_ref.mol
    grids = mf_ref.grids
    dm_ref = mf_ref.make_rdm1()
    dm_test = mf_test.make_rdm1()

    ao = mf_ref._numint.eval_ao(mol, grids.coords, deriv=0)
    rho_ref = mf_ref._numint.eval_rho(mol, ao, dm_ref, xctype='LDA')
    rho_test = mf_test._numint.eval_rho(mol, ao, dm_test, xctype='LDA')

    weights = grids.weights
    delta = rho_test - rho_ref

    nel_ref = np.sum(weights * rho_ref)
    nel_test = np.sum(weights * rho_test)

    L1 = np.sum(weights * np.abs(delta))
    L2 = np.sqrt(np.sum(weights * delta**2))

    rho_denom = np.maximum(rho_ref, rho_floor)
    weighted = np.sum(weights * delta**2 / rho_denom)

    idx_max = np.argmax(np.abs(delta))
    max_delta = delta[idx_max]
    max_delta_r = grids.coords[idx_max]

    return {
        'L1': L1,
        'L2': L2,
        'weighted': weighted,
        'nel_ref': nel_ref,
        'nel_test': nel_test,
        'max_delta': max_delta,
        'max_delta_r': max_delta_r,
    }


# =============================================================================
# Unified Single-Point Runner
# =============================================================================

def run_single_point(
    mol,
    *,
    method: str = "UKS",
    base_xc: str = BASE_XC,
    fisher_cfg: Optional[FisherCfg] = None,
    scc_cfg: Optional[SCCCfg] = None,
    grids_level: int = 4,
    dm0=None,
    max_cycle: int = 150,
    level_shift: float = 0.0,
    damp: float = 0.0,
):
    """
    Run a single-point DFT calculation with optional RIFT corrections.

    Parameters:
        mol: PySCF Mole object
        method: "RKS" or "UKS"
        base_xc: Base XC functional
        fisher_cfg: FisherCfg to enable Fisher correction
        scc_cfg: SCCCfg to enable SCC correction
        grids_level: DFT grid level
        dm0: Initial density matrix
        max_cycle: Max SCF cycles
        level_shift: Level shift for convergence
        damp: Damping factor

    Returns:
        (mf, converged): Mean-field object and convergence status
    """
    from pyscf import dft

    if method.upper() == "UKS":
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)

    mf.grids.level = grids_level
    mf.max_cycle = max_cycle
    mf.diis_space = 12

    if level_shift > 0:
        mf.level_shift = level_shift
    if damp > 0:
        mf.damp = damp

    # Attach corrections in order: base XC -> Fisher -> SCC
    if fisher_cfg is not None:
        FisherModule(fisher_cfg).attach(mf, base_xc=base_xc)
    else:
        mf.xc = base_xc

    # First SCF (without SCC if SCC requested)
    if scc_cfg is not None:
        mf.kernel(dm0=dm0)
        if mf.converged:
            dm0 = mf.make_rdm1()
            SCCModule(scc_cfg).attach(mf)
            mf.kernel(dm0=dm0)
    else:
        mf.kernel(dm0=dm0)

    return mf, mf.converged
