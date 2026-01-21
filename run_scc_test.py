#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy"]
# ///
"""Run just the SCC test - inline version."""

from __future__ import annotations
import numpy as np
from pyscf import gto, dft, scf
from pyscf.dft import libxc
from pyscf.dft.numint import NumInt


def attach_scc_hartree(
    mf,
    *,
    lam: float = 1.0,
    w_power: int = 4,
    z_clip: float = 1e-12,
    rho_floor: float = 1e-12,
):
    """Attach SCC to UKS by overriding get_veff."""
    ni = mf._numint if hasattr(mf, "_numint") else NumInt()
    grids = mf.grids
    mol = mf.mol
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
        if dm_arr.ndim == 2:
            dm_arr = np.asarray(mf.make_rdm1())
        if dm_arr.ndim != 3 or dm_arr.shape[0] != 2:
            return veff

        if grids.coords is None or grids.weights is None:
            grids.build(with_non0tab=True)

        ao = ni.eval_ao(mol, grids.coords, deriv=1)
        ao0, aox, aoy, aoz = ao[0], ao[1], ao[2], ao[3]

        rho_a_full = ni.eval_rho(mol, ao, dm_arr[0], xctype="GGA", hermi=hermi)
        rho_b_full = ni.eval_rho(mol, ao, dm_arr[1], xctype="GGA", hermi=hermi)
        rho_a, rho_b = rho_a_full[0], rho_b_full[0]
        rho_tot = rho_a + rho_b

        mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ

        def tau_from_spin(spin_idx: int):
            C = mo_coeff[spin_idx]
            occ = mo_occ[spin_idx]
            occ_mask = occ > 1e-12
            Cocc = C[:, occ_mask]
            occv = occ[occ_mask]
            gx = aox @ Cocc
            gy = aoy @ Cocc
            gz = aoz @ Cocc
            g2 = (gx * gx + gy * gy + gz * gz) * occv
            return 0.5 * np.sum(g2, axis=1)

        tau_tot = tau_from_spin(0) + tau_from_spin(1)
        grad_tot = rho_a_full[1:4] + rho_b_full[1:4]
        sigma_tot = np.einsum("ig,ig->g", grad_tot, grad_tot)
        rho_eff = np.maximum(rho_tot, rho_floor)
        tauW = sigma_tot / (8.0 * rho_eff)

        tau_eff = np.maximum(tau_tot, z_clip)
        z = np.clip(tauW / tau_eff, 0.0, 1.0)
        w = gate_from_z(z)

        dm_tot = dm_arr[0] + dm_arr[1]
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


def _h2plus_single_point(R, mode, scc_lam=0.1):
    """Run single H₂⁺ calculation."""
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
        return mf.e_tot, mf.converged

    mf = dft.UKS(mol)
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 150
    if R > 2.0:
        mf.level_shift = 0.2

    if mode == "pbe":
        mf.xc = "GGA_X_PBE,GGA_C_PBE"
    elif mode == "lcwpbe":
        mf.xc = "LRC_WPBE"
        mf.omega = 0.4
    elif mode == "scc":
        mf.xc = "GGA_X_PBE,GGA_C_PBE"
        mf.kernel()
        if mf.converged:
            dm0 = mf.make_rdm1()
            attach_scc_hartree(mf, lam=scc_lam, w_power=4)
            mf.kernel(dm0=dm0)
        return mf.e_tot, mf.converged
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mf.kernel()
    return mf.e_tot, mf.converged


def run_h2plus_scc_test():
    """H₂⁺ SCC test."""
    print("\n" + "=" * 80)
    print("H₂⁺ SCC (Self-Coupling Cancellation) TEST")
    print("=" * 80)
    print("Testing whether SCC moves binding curve in same direction as UHF/LC-ωPBE.")
    print("Using small λ values to avoid over-correction.\n")

    distances = np.array([0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    lam_values = [0.05, 0.1]

    results = {"pbe": [], "uhf": [], "lcwpbe": []}
    for lam in lam_values:
        results[f"scc_{lam}"] = []

    print(f"{'R(Å)':>6}  {'PBE':>12}  {'UHF':>12}  {'LC-ωPBE':>12}", end="")
    for lam in lam_values:
        print(f"  {'SCC λ='+str(lam):>12}", end="")
    print("  conv")
    print("-" * (55 + 14 * len(lam_values)))

    for R in distances:
        row = f"{R:6.3f}"
        conv_str = ""

        e_pbe, c = _h2plus_single_point(R, "pbe")
        results["pbe"].append((R, e_pbe, c))
        row += f"  {e_pbe:12.8f}"
        conv_str += "✓" if c else "✗"

        e_uhf, c = _h2plus_single_point(R, "uhf")
        results["uhf"].append((R, e_uhf, c))
        row += f"  {e_uhf:12.8f}"
        conv_str += "✓" if c else "✗"

        e_lc, c = _h2plus_single_point(R, "lcwpbe")
        results["lcwpbe"].append((R, e_lc, c))
        row += f"  {e_lc:12.8f}"
        conv_str += "✓" if c else "✗"

        for lam in lam_values:
            try:
                e_scc, c = _h2plus_single_point(R, "scc", scc_lam=lam)
                results[f"scc_{lam}"].append((R, e_scc, c))
                row += f"  {e_scc:12.8f}"
                conv_str += "✓" if c else "✗"
            except Exception:
                results[f"scc_{lam}"].append((R, np.nan, False))
                row += f"  {'ERROR':>12}"
                conv_str += "E"

        print(row + f"  {conv_str}")

    # Binding energies
    print(f"\n--- Binding Energy (E - E_∞) in mHa ---")
    e_inf = {key: results[key][-1][1] for key in results}

    for i, R in enumerate(distances):
        be = {key: (results[key][i][1] - e_inf[key]) * 1000 for key in results}
        row = f"{R:6.3f}  {be['pbe']:+10.2f}  {be['uhf']:+10.2f}  {be['lcwpbe']:+10.2f}"
        d_uhf = be["uhf"] - be["pbe"]
        row += f"  {d_uhf:+10.2f}"
        for lam in lam_values:
            d_scc = be[f"scc_{lam}"] - be["pbe"]
            row += f"  {d_scc:+10.2f}"
        print(row)

    # Summary at R=2.0
    idx_2 = list(distances).index(2.0)
    be_pbe = (results["pbe"][idx_2][1] - e_inf["pbe"]) * 1000
    be_uhf = (results["uhf"][idx_2][1] - e_inf["uhf"]) * 1000
    d_uhf = be_uhf - be_pbe

    print(f"\nAt R = 2.0 Å:")
    print(f"  Δ(UHF-PBE) = {d_uhf:+.2f} mHa")
    for lam in lam_values:
        be_scc = (results[f"scc_{lam}"][idx_2][1] - e_inf[f"scc_{lam}"]) * 1000
        d_scc = be_scc - be_pbe
        sign = "✓ CORRECT" if d_scc * d_uhf > 0 else "✗ WRONG"
        print(f"  Δ(SCC λ={lam}-PBE) = {d_scc:+.2f} mHa ({sign})")

    return results


if __name__ == "__main__":
    run_h2plus_scc_test()
