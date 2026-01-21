#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy"]
# ///
"""
RIFT Composability Test: Fisher + SCC on H₂⁺ and H₂

Tests whether Fisher (geometry/bonding) and SCC (self-interaction)
correction channels compose cleanly or interfere.

Decision criteria:
  - Additivity: ε(R) = ΔE_FS - (ΔE_F + ΔE_S) should be small
  - No sign conflicts: SCC effect should remain negative at mid-R for H₂⁺
"""

from __future__ import annotations
import numpy as np
from pyscf import gto, dft, scf
from pyscf.dft import libxc
from pyscf.dft.numint import NumInt
import csv
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# RIFT XC Functional (Fisher + Harmonizer)
# =============================================================================

def make_rift_xc(
    *,
    base_xc: str = "GGA_X_PBE,GGA_C_PBE",
    alpha: float = 0.0,
    rho_floor: float = 1e-7,
    harmonize: bool = True,
    rho_core: float = 1.0,
    core_power: float = 4.0,
):
    """RIFT XC with Fisher + Harmonizer gates."""

    def xc_callable(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        rho_arr = np.asarray(rho, dtype=float)
        spin_int = int(np.asarray(spin).ravel()[0]) if hasattr(spin, '__iter__') else int(spin)

        # Handle UKS
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

                # Harmonizer gate f(s)
                f_s = 1.0
                df_term = 0.0
                if harmonize:
                    c_s = 2.0 * (3.0 * np.pi**2) ** (1.0/3.0)
                    grad_norm = np.sqrt(np.maximum(sigma_tot, 1e-30))
                    s = grad_norm / (c_s * rf ** (4.0/3.0))
                    s2 = s * s
                    f_s = s2 / (1.0 + s2)
                    df_term = (8.0/3.0) * s2 / (1.0 + s2) ** 2

                # Core suppression gate g(ρ)
                g_rho = 1.0
                dg_term = 0.0
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

                # Apply to UKS vxc
                vrho = np.asarray(vxc[0]).copy()
                vsigma = np.asarray(vxc[1]).copy()
                if vrho.ndim == 2:
                    vrho[:, 0] += vrho_add
                    vrho[:, 1] += vrho_add
                else:
                    vrho += vrho_add
                vsigma_arr = np.asarray(vsigma_add)
                if vsigma.ndim == 2:
                    vsigma[:, 0] += vsigma_arr
                    vsigma[:, 1] += 2.0 * vsigma_arr
                    vsigma[:, 2] += vsigma_arr
                else:
                    vsigma += vsigma_arr
                rest = list(vxc[2:]) if len(vxc) > 2 else []
                vxc = (vrho, vsigma, *rest)

            return exc, vxc, None, None

        # RKS path
        if rho_arr.ndim == 1:
            rho_g = rho_arr
            gga_rho = np.vstack([rho_g, np.zeros_like(rho_g),
                                 np.zeros_like(rho_g), np.zeros_like(rho_g)])
            has_grad = False
        elif rho_arr.ndim == 2 and rho_arr.shape[0] < 4:
            rho_g = rho_arr[0]
            gga_rho = np.vstack([rho_g, np.zeros_like(rho_g),
                                 np.zeros_like(rho_g), np.zeros_like(rho_g)])
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

            # Add to vxc
            if isinstance(vxc, (tuple, list)):
                vxc0 = np.asarray(vxc[0]) + vrho_add
                vxc1 = np.asarray(vxc[1]) + vsigma_add
                rest = list(vxc[2:])
                vxc = (vxc0, vxc1, *rest)
            else:
                v = np.asarray(vxc).copy()
                v[0] += vrho_add
                v[1] += vsigma_add
                vxc = v

        return exc, vxc, None, None

    return xc_callable


# =============================================================================
# SCC (Self-Coupling Cancellation)
# =============================================================================

def attach_scc_hartree(mf, *, lam: float = 0.1, w_power: int = 4,
                       z_clip: float = 1e-12, rho_floor: float = 1e-12):
    """Attach SCC correction to UKS/RKS by overriding get_veff."""
    ni = mf._numint if hasattr(mf, "_numint") else NumInt()
    grids = mf.grids
    get_veff_orig = mf.get_veff

    def gate_from_z(z):
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

        # Handle RKS (2D) vs UKS (3D)
        is_uks = dm_arr.ndim == 3 and dm_arr.shape[0] == 2

        if dm_arr.ndim == 2:
            # RKS: use stored MO info if available
            if not hasattr(mf, 'mo_coeff') or mf.mo_coeff is None:
                return veff
            dm_arr_work = dm_arr
        elif is_uks:
            dm_arr_work = dm_arr
        else:
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

        # Compute tau from orbitals
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

        J = mf.get_j(mol, dm_tot)

        wt = grids.weights
        rho_weighted = rho_tot * wt
        w_avg = np.sum(w * rho_weighted) / np.maximum(np.sum(rho_weighted), 1e-30)

        scale_factor = lam * w_avg
        delta_J = -scale_factor * J

        veff_arr = np.asarray(veff)
        if veff_arr.ndim == 3:
            veff_new = veff_arr.copy()
            veff_new[0] = veff_new[0] + delta_J
            veff_new[1] = veff_new[1] + delta_J
        else:
            veff_new = veff_arr + delta_J

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
# Single-Point Calculation
# =============================================================================

@dataclass
class CalcResult:
    R: float
    mode: str
    E: float
    converged: bool
    homo: Optional[float] = None
    lumo: Optional[float] = None


def run_h2plus(R: float, mode: str, alpha: float = 0.0, lam: float = 0.0) -> CalcResult:
    """H₂⁺ single point. Modes: pbe, fisher, scc, both, uhf"""
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R}",
        basis="def2-svp",
        charge=1,
        spin=1,
        verbose=0,
    )

    if mode == "uhf":
        mf = scf.UHF(mol)
        mf.diis_space = 12
        mf.max_cycle = 150
        if R > 2.0:
            mf.level_shift = 0.2
        mf.kernel()
        return CalcResult(R=R, mode=mode, E=mf.e_tot, converged=mf.converged)

    mf = dft.UKS(mol)
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 150
    if R > 2.0:
        mf.level_shift = 0.2
        mf.damp = 0.3

    use_fisher = mode in ("fisher", "both")
    use_scc = mode in ("scc", "both")

    if use_fisher:
        mf.define_xc_(make_rift_xc(alpha=alpha, harmonize=True, rho_core=1.0), xctype="GGA")
    else:
        mf.xc = "GGA_X_PBE,GGA_C_PBE"

    # First SCF (PBE or Fisher)
    mf.kernel()

    if use_scc and mf.converged:
        dm0 = mf.make_rdm1()
        attach_scc_hartree(mf, lam=lam, w_power=4)
        mf.kernel(dm0=dm0)

    return CalcResult(R=R, mode=mode, E=mf.e_tot, converged=mf.converged)


def run_h2(R: float, mode: str, alpha: float = 0.0, lam: float = 0.0,
           use_uks: bool = False) -> CalcResult:
    """H₂ single point. Modes: pbe, fisher, scc, both, hf"""
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R}",
        basis="def2-svp",
        charge=0,
        spin=0,
        verbose=0,
    )

    if mode == "hf":
        if use_uks:
            mf = scf.UHF(mol)
        else:
            mf = scf.RHF(mol)
        mf.diis_space = 12
        mf.max_cycle = 150
        if R > 2.0:
            mf.level_shift = 0.2
        mf.kernel()
        return CalcResult(R=R, mode=mode, E=mf.e_tot, converged=mf.converged)

    if use_uks:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)

    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 150
    if R > 2.0:
        mf.level_shift = 0.2
        mf.damp = 0.3

    use_fisher = mode in ("fisher", "both")
    use_scc = mode in ("scc", "both")

    if use_fisher:
        mf.define_xc_(make_rift_xc(alpha=alpha, harmonize=True, rho_core=1.0), xctype="GGA")
    else:
        mf.xc = "GGA_X_PBE,GGA_C_PBE"

    mf.kernel()

    if use_scc and mf.converged:
        dm0 = mf.make_rdm1()
        attach_scc_hartree(mf, lam=lam, w_power=4)
        mf.kernel(dm0=dm0)

    return CalcResult(R=R, mode=mode, E=mf.e_tot, converged=mf.converged)


# =============================================================================
# Calibration
# =============================================================================

def calibrate_alpha(system: str, R_ref: float, target_dE_mHa: float = 20.0) -> float:
    """Find α that gives target ΔE at reference geometry."""
    print(f"\n--- Calibrating α for {system} at R={R_ref} Å ---")
    print(f"Target: ΔE ≈ {target_dE_mHa:+.1f} mHa")

    if system == "h2plus":
        e_pbe = run_h2plus(R_ref, "pbe").E
    else:
        e_pbe = run_h2(R_ref, "pbe").E

    alpha_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    de_vals = {}

    for a in alpha_grid:
        if system == "h2plus":
            e = run_h2plus(R_ref, "fisher", alpha=a).E
        else:
            e = run_h2(R_ref, "fisher", alpha=a).E
        de = (e - e_pbe) * 1000
        de_vals[a] = de
        print(f"  α={a:.3f}: ΔE = {de:+.2f} mHa")

    # Interpolate to target
    alphas = np.array(list(de_vals.keys()))
    des = np.array(list(de_vals.values()))
    alpha_cal = np.interp(target_dE_mHa, des, alphas)

    print(f"  => Calibrated α = {alpha_cal:.4f}")
    return alpha_cal


def calibrate_lambda(system: str, R_ref: float, target_dE_mHa: float = -10.0) -> float:
    """Find λ that gives target ΔE at reference geometry."""
    print(f"\n--- Calibrating λ for {system} at R={R_ref} Å ---")
    print(f"Target: ΔE ≈ {target_dE_mHa:+.1f} mHa")

    if system == "h2plus":
        e_pbe = run_h2plus(R_ref, "pbe").E
    else:
        e_pbe = run_h2(R_ref, "pbe").E

    lam_grid = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    de_vals = {}

    for l in lam_grid:
        if system == "h2plus":
            e = run_h2plus(R_ref, "scc", lam=l).E
        else:
            e = run_h2(R_ref, "scc", lam=l).E
        de = (e - e_pbe) * 1000
        de_vals[l] = de
        print(f"  λ={l:.3f}: ΔE = {de:+.2f} mHa")

    # Interpolate to target
    lams = np.array(list(de_vals.keys()))
    des = np.array(list(de_vals.values()))
    lam_cal = np.interp(target_dE_mHa, des, lams)

    print(f"  => Calibrated λ = {lam_cal:.4f}")
    return lam_cal


# =============================================================================
# Main Test
# =============================================================================

def run_composability_test():
    """Run full composability test for Fisher + SCC."""

    print("=" * 80)
    print("RIFT COMPOSABILITY TEST: Fisher + SCC")
    print("=" * 80)
    print()
    print("Testing whether Fisher (geometry) and SCC (self-interaction)")
    print("correction channels compose additively or interfere.")
    print()

    # =========================================================================
    # H₂⁺ Test
    # =========================================================================
    print("\n" + "=" * 80)
    print("SYSTEM: H₂⁺ (UKS) - Primary SIE test")
    print("=" * 80)

    R_ref_h2plus = 2.0  # Mid-R where SIE is prominent

    # Calibrate parameters
    alpha_h2plus = calibrate_alpha("h2plus", R_ref_h2plus, target_dE_mHa=15.0)
    lam_h2plus = calibrate_lambda("h2plus", R_ref_h2plus, target_dE_mHa=-10.0)

    print(f"\nUsing: α = {alpha_h2plus:.4f}, λ = {lam_h2plus:.4f}")

    # Dissociation scan
    distances_h2plus = np.array([0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    modes = ["pbe", "fisher", "scc", "both"]

    results_h2plus = {m: [] for m in modes}
    results_h2plus["uhf"] = []

    print(f"\n--- H₂⁺ Dissociation Scan ---")
    print(f"{'R(Å)':>6}  {'PBE':>12}  {'Fisher':>12}  {'SCC':>12}  {'Both':>12}  {'UHF':>12}  conv")
    print("-" * 85)

    for R in distances_h2plus:
        row = f"{R:6.3f}"
        conv = ""

        for mode in modes:
            res = run_h2plus(R, mode, alpha=alpha_h2plus, lam=lam_h2plus)
            results_h2plus[mode].append(res)
            row += f"  {res.E:12.8f}"
            conv += "✓" if res.converged else "✗"

        # UHF reference
        res_uhf = run_h2plus(R, "uhf")
        results_h2plus["uhf"].append(res_uhf)
        row += f"  {res_uhf.E:12.8f}"
        conv += "✓" if res_uhf.converged else "✗"

        print(row + f"  {conv}")

    # Compute deltas and epsilon
    print(f"\n--- H₂⁺ Composability Analysis ---")
    print(f"{'R(Å)':>6}  {'ΔE_F':>10}  {'ΔE_S':>10}  {'ΔE_FS':>10}  {'ε(R)':>10}  {'ε/(|F|+|S|)':>12}  {'SCC sign':>10}")
    print("-" * 85)

    h2plus_data = []
    for i, R in enumerate(distances_h2plus):
        e_pbe = results_h2plus["pbe"][i].E
        e_f = results_h2plus["fisher"][i].E
        e_s = results_h2plus["scc"][i].E
        e_fs = results_h2plus["both"][i].E
        e_uhf = results_h2plus["uhf"][i].E

        dE_F = (e_f - e_pbe) * 1000
        dE_S = (e_s - e_pbe) * 1000
        dE_FS = (e_fs - e_pbe) * 1000
        epsilon = dE_FS - (dE_F + dE_S)

        denom = abs(dE_F) + abs(dE_S)
        eps_rel = epsilon / denom if denom > 0.1 else 0.0

        # Check SCC sign (should be negative for SIE correction at mid-R)
        scc_sign = "✓ neg" if dE_S < 0 else "✗ pos"
        if dE_FS < dE_F:  # Combined should still show SCC effect
            both_sign = "✓"
        else:
            both_sign = "✗ conflict"

        print(f"{R:6.3f}  {dE_F:+10.2f}  {dE_S:+10.2f}  {dE_FS:+10.2f}  {epsilon:+10.2f}  {eps_rel:+12.2%}  {scc_sign}")

        h2plus_data.append({
            'R': R, 'E_pbe': e_pbe, 'E_fisher': e_f, 'E_scc': e_s, 'E_both': e_fs, 'E_uhf': e_uhf,
            'dE_F': dE_F, 'dE_S': dE_S, 'dE_FS': dE_FS, 'epsilon': epsilon,
            'conv_pbe': results_h2plus["pbe"][i].converged,
            'conv_fisher': results_h2plus["fisher"][i].converged,
            'conv_scc': results_h2plus["scc"][i].converged,
            'conv_both': results_h2plus["both"][i].converged,
        })

    # =========================================================================
    # H₂ Test (RKS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SYSTEM: H₂ (RKS) - Geometry/bonding test")
    print("=" * 80)

    R_ref_h2 = 0.74  # Near equilibrium

    # Calibrate parameters
    alpha_h2 = calibrate_alpha("h2", R_ref_h2, target_dE_mHa=15.0)
    lam_h2 = calibrate_lambda("h2", R_ref_h2, target_dE_mHa=-10.0)

    print(f"\nUsing: α = {alpha_h2:.4f}, λ = {lam_h2:.4f}")

    # Dissociation scan
    distances_h2 = np.array([0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0])

    results_h2 = {m: [] for m in modes}
    results_h2["hf"] = []

    print(f"\n--- H₂ Dissociation Scan ---")
    print(f"{'R(Å)':>6}  {'PBE':>12}  {'Fisher':>12}  {'SCC':>12}  {'Both':>12}  {'HF':>12}  conv")
    print("-" * 85)

    for R in distances_h2:
        row = f"{R:6.3f}"
        conv = ""

        for mode in modes:
            res = run_h2(R, mode, alpha=alpha_h2, lam=lam_h2, use_uks=False)
            results_h2[mode].append(res)
            row += f"  {res.E:12.8f}"
            conv += "✓" if res.converged else "✗"

        # HF reference
        res_hf = run_h2(R, "hf", use_uks=False)
        results_h2["hf"].append(res_hf)
        row += f"  {res_hf.E:12.8f}"
        conv += "✓" if res_hf.converged else "✗"

        print(row + f"  {conv}")

    # Compute deltas and epsilon
    print(f"\n--- H₂ Composability Analysis ---")
    print(f"{'R(Å)':>6}  {'ΔE_F':>10}  {'ΔE_S':>10}  {'ΔE_FS':>10}  {'ε(R)':>10}  {'ε/(|F|+|S|)':>12}")
    print("-" * 70)

    h2_data = []
    for i, R in enumerate(distances_h2):
        e_pbe = results_h2["pbe"][i].E
        e_f = results_h2["fisher"][i].E
        e_s = results_h2["scc"][i].E
        e_fs = results_h2["both"][i].E
        e_hf = results_h2["hf"][i].E

        dE_F = (e_f - e_pbe) * 1000
        dE_S = (e_s - e_pbe) * 1000
        dE_FS = (e_fs - e_pbe) * 1000
        epsilon = dE_FS - (dE_F + dE_S)

        denom = abs(dE_F) + abs(dE_S)
        eps_rel = epsilon / denom if denom > 0.1 else 0.0

        print(f"{R:6.3f}  {dE_F:+10.2f}  {dE_S:+10.2f}  {dE_FS:+10.2f}  {epsilon:+10.2f}  {eps_rel:+12.2%}")

        h2_data.append({
            'R': R, 'E_pbe': e_pbe, 'E_fisher': e_f, 'E_scc': e_s, 'E_both': e_fs, 'E_hf': e_hf,
            'dE_F': dE_F, 'dE_S': dE_S, 'dE_FS': dE_FS, 'epsilon': epsilon,
            'conv_pbe': results_h2["pbe"][i].converged,
            'conv_fisher': results_h2["fisher"][i].converged,
            'conv_scc': results_h2["scc"][i].converged,
            'conv_both': results_h2["both"][i].converged,
        })

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPOSABILITY SUMMARY")
    print("=" * 80)

    # H₂⁺ analysis
    eps_h2plus = [d['epsilon'] for d in h2plus_data]
    dE_sum_h2plus = [abs(d['dE_F']) + abs(d['dE_S']) for d in h2plus_data]
    eps_rel_h2plus = [e/s if s > 0.1 else 0 for e, s in zip(eps_h2plus, dE_sum_h2plus)]

    print(f"\nH₂⁺:")
    print(f"  ε(R) range: {min(eps_h2plus):+.2f} to {max(eps_h2plus):+.2f} mHa")
    print(f"  |ε|/(|ΔF|+|ΔS|) max: {max(abs(e) for e in eps_rel_h2plus):.1%}")

    # Check SCC sign preservation
    scc_signs_ok = all(d['dE_S'] < 0 for d in h2plus_data if d['R'] >= 1.5)
    both_preserves_scc = all(d['dE_FS'] < d['dE_F'] for d in h2plus_data if d['R'] >= 1.5)

    print(f"  SCC negative at mid-R: {'✓ YES' if scc_signs_ok else '✗ NO'}")
    print(f"  Combined preserves SCC: {'✓ YES' if both_preserves_scc else '✗ NO'}")

    # H₂ analysis
    eps_h2 = [d['epsilon'] for d in h2_data]
    dE_sum_h2 = [abs(d['dE_F']) + abs(d['dE_S']) for d in h2_data]
    eps_rel_h2 = [e/s if s > 0.1 else 0 for e, s in zip(eps_h2, dE_sum_h2)]

    print(f"\nH₂:")
    print(f"  ε(R) range: {min(eps_h2):+.2f} to {max(eps_h2):+.2f} mHa")
    print(f"  |ε|/(|ΔF|+|ΔS|) max: {max(abs(e) for e in eps_rel_h2):.1%}")

    # Overall verdict
    print(f"\n" + "-" * 40)
    max_eps_rel = max(max(abs(e) for e in eps_rel_h2plus), max(abs(e) for e in eps_rel_h2))

    if max_eps_rel < 0.2 and scc_signs_ok and both_preserves_scc:
        print("VERDICT: ✓ COMPOSABLE")
        print("Fisher and SCC channels are approximately additive")
        print("and do not conflict in their correction effects.")
    elif max_eps_rel < 0.5 and scc_signs_ok:
        print("VERDICT: ⚠ WEAKLY COMPOSABLE")
        print("Some non-additivity present, but no sign conflicts.")
        print("Consider gate adjustments if using large α/λ.")
    else:
        print("VERDICT: ✗ NOT COMPOSABLE")
        print("Significant interference between Fisher and SCC.")
        print("Investigate gate overlap before combining.")

    # Write CSV
    with open('composability_h2plus.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=h2plus_data[0].keys())
        writer.writeheader()
        writer.writerows(h2plus_data)

    with open('composability_h2.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=h2_data[0].keys())
        writer.writeheader()
        writer.writerows(h2_data)

    print(f"\nResults written to composability_h2plus.csv and composability_h2.csv")

    return h2plus_data, h2_data


if __name__ == "__main__":
    run_composability_test()
