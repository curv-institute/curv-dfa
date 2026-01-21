"""
RIFT Experiments: Test harnesses for Fisher + SCC corrections.

This module contains all experimental runners for validating
and characterizing the RIFT functional corrections.
"""
from __future__ import annotations

import numpy as np
from pyscf import gto, dft, scf

from rift_functionals import (
    BASE_XC,
    FisherCfg,
    SCCCfg,
    FisherModule,
    SCCModule,
    make_rift_xc,
    attach_scc_hartree,
    compute_rift_energy,
    compute_radial_fisher_decomposition,
    get_orbital_energies,
    compute_density_diagnostics,
    run_single_point,
)


# =============================================================================
# Validation
# =============================================================================

def run_validation():
    """Validate that α=0 matches native PBE exactly."""
    print("\n" + "=" * 80)
    print("VALIDATION: α=0 should match native PBE")
    print("=" * 80)

    mol = gto.M(
        atom="""
        O 0.000000  0.000000  0.000000
        H 0.000000 -0.757000  0.587000
        H 0.000000  0.757000  0.587000
        """,
        basis="def2-svp",
        verbose=0,
    )

    # Native PBE
    mf_pbe = dft.RKS(mol)
    mf_pbe.xc = 'PBE'
    mf_pbe.grids.level = 4
    mf_pbe.kernel()
    print(f"Native PBE:     E = {mf_pbe.e_tot:.10f}")

    # RIFT with α=0, β=0
    mf_rift = dft.RKS(mol)
    mf_rift.grids.level = 4
    mf_rift.define_xc_(make_rift_xc(alpha=0.0, beta=0.0), xctype='GGA')
    mf_rift.kernel()
    print(f"RIFT (α=0,β=0): E = {mf_rift.e_tot:.10f}")

    de = (mf_rift.e_tot - mf_pbe.e_tot) * 1000
    print(f"Difference:     ΔE = {de:.6f} mHa")

    if abs(de) < 0.1:
        print("✓ PASS: α=0 reproduces native PBE")
    else:
        print("✗ FAIL: α=0 does NOT match native PBE")

    return abs(de) < 0.1


# =============================================================================
# H₂⁺ Single-Point Helper
# =============================================================================

def _h2plus_single_point(R, mode, alpha=0.0, rho_core=1.0, omega=0.4, scc_lam=0.1):
    """
    Run single H₂⁺ calculation.

    Modes: pbe, uhf, pbe0, lcwpbe, fisher, scc, both
    """
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R}",
        basis="def2-svp",
        charge=1,
        spin=1,
        verbose=0,
    )

    # UHF reference (exact for 1 electron)
    if mode == "uhf":
        mf = scf.UHF(mol)
        mf.diis_space = 12
        mf.max_cycle = 150
        if R > 2.0:
            mf.level_shift = 0.2
            mf.damp = 0.3
        mf.kernel()
        return mf.e_tot, mf.converged

    # DFT paths
    mf = dft.UKS(mol)
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 150
    if R > 2.0:
        mf.level_shift = 0.2
        mf.damp = 0.3

    if mode == "pbe":
        mf.xc = BASE_XC
    elif mode == "pbe0":
        mf.xc = "PBE0"
    elif mode == "lcwpbe":
        mf.xc = "LRC_WPBE"
        mf.omega = omega
    elif mode == "fisher":
        FisherModule(FisherCfg(alpha=alpha, rho_core=rho_core)).attach(mf)
    elif mode == "scc":
        mf.xc = BASE_XC
        mf.kernel()
        if mf.converged:
            dm0 = mf.make_rdm1()
            SCCModule(SCCCfg(lam=scc_lam)).attach(mf)
            mf.kernel(dm0=dm0)
        return mf.e_tot, mf.converged
    elif mode == "both":
        FisherModule(FisherCfg(alpha=alpha, rho_core=rho_core)).attach(mf)
        mf.kernel()
        if mf.converged:
            dm0 = mf.make_rdm1()
            SCCModule(SCCCfg(lam=scc_lam)).attach(mf)
            mf.kernel(dm0=dm0)
        return mf.e_tot, mf.converged
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mf.kernel()
    return mf.e_tot, mf.converged


# =============================================================================
# H₂ Single-Point Helper
# =============================================================================

def _h2_single_point(R, mode, alpha=0.0, rho_core=1.0, scc_lam=0.1, use_uks=False):
    """
    Run single H₂ calculation.

    Modes: pbe, hf, fisher, scc, both
    """
    mol = gto.M(
        atom=f"H 0 0 0; H 0 0 {R}",
        basis="def2-svp",
        charge=0,
        spin=0,
        verbose=0,
    )

    # HF reference
    if mode == "hf":
        mf = scf.UHF(mol) if use_uks else scf.RHF(mol)
        mf.diis_space = 12
        mf.max_cycle = 150
        if R > 2.0:
            mf.level_shift = 0.2
        mf.kernel()
        return mf.e_tot, mf.converged

    # DFT
    mf = dft.UKS(mol) if use_uks else dft.RKS(mol)
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 150
    if R > 2.0:
        mf.level_shift = 0.2
        mf.damp = 0.3

    if mode == "pbe":
        mf.xc = BASE_XC
    elif mode == "fisher":
        FisherModule(FisherCfg(alpha=alpha, rho_core=rho_core)).attach(mf)
    elif mode == "scc":
        mf.xc = BASE_XC
        mf.kernel()
        if mf.converged:
            dm0 = mf.make_rdm1()
            SCCModule(SCCCfg(lam=scc_lam)).attach(mf)
            mf.kernel(dm0=dm0)
        return mf.e_tot, mf.converged
    elif mode == "both":
        FisherModule(FisherCfg(alpha=alpha, rho_core=rho_core)).attach(mf)
        mf.kernel()
        if mf.converged:
            dm0 = mf.make_rdm1()
            SCCModule(SCCCfg(lam=scc_lam)).attach(mf)
            mf.kernel(dm0=dm0)
        return mf.e_tot, mf.converged
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mf.kernel()
    return mf.e_tot, mf.converged


# =============================================================================
# H₂⁺ SIE Validation
# =============================================================================

def run_h2plus_sie_validation():
    """Validate harness can detect SIE using UHF and LC-ωPBE controls."""
    print("\n" + "=" * 80)
    print("H₂⁺ SIE HARNESS VALIDATION")
    print("=" * 80)
    print("Controls: UHF (exact, no SIE) and LC-ωPBE (RSH, reduced SIE)")
    print("If these show different mid-R behavior than PBE, harness is SIE-sensitive.\n")

    distances = np.linspace(0.7, 5.0, 18)
    modes = ["pbe", "uhf", "lcwpbe"]

    print(f"{'R(Å)':>6}", end="")
    for m in modes:
        print(f"  {m:>12}", end="")
    print("  conv")
    print("-" * 55)

    results = {m: [] for m in modes}

    for R in distances:
        row = f"{R:6.3f}"
        conv_all = ""
        for m in modes:
            try:
                e, conv = _h2plus_single_point(R, m)
                results[m].append((R, e, conv))
                row += f"  {e:12.8f}"
                conv_all += "✓" if conv else "✗"
            except Exception:
                results[m].append((R, np.nan, False))
                row += f"  {'ERROR':>12}"
                conv_all += "E"
        print(row + f"  {conv_all}")

    # Binding energies
    print(f"\n--- Binding Energy (E - E_∞) in mHa ---")
    print(f"{'R(Å)':>6}  {'PBE':>10}  {'UHF':>10}  {'LC-ωPBE':>10}  {'Δ(UHF-PBE)':>12}  {'Δ(LC-PBE)':>12}")
    print("-" * 70)

    e_inf = {m: results[m][-1][1] for m in modes}

    for i, R in enumerate(distances):
        be = {m: (results[m][i][1] - e_inf[m]) * 1000 for m in modes}
        d_uhf = be["uhf"] - be["pbe"]
        d_lc = be["lcwpbe"] - be["pbe"]
        print(f"{R:6.3f}  {be['pbe']:+10.2f}  {be['uhf']:+10.2f}  {be['lcwpbe']:+10.2f}  {d_uhf:+12.2f}  {d_lc:+12.2f}")

    print("\n--- Interpretation ---")
    print("If Δ(UHF-PBE) and Δ(LC-PBE) are significantly negative at mid-R (1.5-3Å),")
    print("the harness correctly detects SIE reduction.")

    return results


# =============================================================================
# H₂⁺ SCC Test
# =============================================================================

def run_h2plus_scc_test():
    """Test SCC sign: should move binding in same direction as UHF."""
    print("\n" + "=" * 80)
    print("H₂⁺ SCC (Self-Coupling Cancellation) TEST")
    print("=" * 80)
    print("Testing whether SCC moves binding curve in same direction as UHF/LC-ωPBE.")
    print("Using small λ values to avoid over-correction.\n")

    distances = np.linspace(0.7, 5.0, 18)
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

        for mode in ["pbe", "uhf", "lcwpbe"]:
            e, c = _h2plus_single_point(R, mode)
            results[mode].append((R, e, c))
            row += f"  {e:12.8f}"
            conv_str += "✓" if c else "✗"

        for lam in lam_values:
            try:
                e, c = _h2plus_single_point(R, "scc", scc_lam=lam)
                results[f"scc_{lam}"].append((R, e, c))
                row += f"  {e:12.8f}"
                conv_str += "✓" if c else "✗"
            except Exception:
                results[f"scc_{lam}"].append((R, np.nan, False))
                row += f"  {'ERROR':>12}"
                conv_str += "E"

        print(row + f"  {conv_str}")

    # Binding energies
    print(f"\n--- Binding Energy (E - E_∞) in mHa ---")
    e_inf = {k: results[k][-1][1] for k in results}

    for i, R in enumerate(distances):
        be = {k: (results[k][i][1] - e_inf[k]) * 1000 for k in results}
        row = f"{R:6.3f}  {be['pbe']:+10.2f}  {be['uhf']:+10.2f}  {be['lcwpbe']:+10.2f}"
        for lam in lam_values:
            row += f"  {be[f'scc_{lam}']:+10.2f}"
        print(row)

    # Summary
    idx_2 = np.argmin(np.abs(distances - 2.0))
    R_test = distances[idx_2]
    be_pbe = (results["pbe"][idx_2][1] - e_inf["pbe"]) * 1000
    be_uhf = (results["uhf"][idx_2][1] - e_inf["uhf"]) * 1000

    print(f"\nAt R = {R_test:.2f} Å:")
    print(f"  Δ(UHF-PBE) = {be_uhf - be_pbe:+.2f} mHa")
    for lam in lam_values:
        be_scc = (results[f"scc_{lam}"][idx_2][1] - e_inf[f"scc_{lam}"]) * 1000
        d_scc = be_scc - be_pbe
        sign = "✓ CORRECT" if d_scc < 0 else "✗ WRONG"
        print(f"  Δ(SCC λ={lam}-PBE) = {d_scc:+.2f} mHa ({sign})")

    return results


# =============================================================================
# Composability Test (Fisher + SCC)
# =============================================================================

def run_composability_test(
    alpha_h2plus: float = 0.006,
    lam_h2plus: float = 0.2,
    alpha_h2: float = 0.005,
    lam_h2: float = 0.2,
):
    """
    Test whether Fisher and SCC compose additively.

    Decision criteria:
      - ε(R) = ΔE_FS - (ΔE_F + ΔE_S) should be small (<5% of total)
      - No sign conflicts (SCC stays negative at mid-R for H₂⁺)
    """
    print("\n" + "=" * 80)
    print("COMPOSABILITY TEST: Fisher + SCC")
    print("=" * 80)

    # H₂⁺
    print(f"\n--- H₂⁺ (α={alpha_h2plus}, λ={lam_h2plus}) ---")
    distances = np.array([0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])

    print(f"{'R(Å)':>6}  {'ΔE_F':>10}  {'ΔE_S':>10}  {'ΔE_FS':>10}  {'ε(R)':>10}  {'ε%':>8}")
    print("-" * 65)

    for R in distances:
        e_pbe, _ = _h2plus_single_point(R, "pbe")
        e_f, _ = _h2plus_single_point(R, "fisher", alpha=alpha_h2plus)
        e_s, _ = _h2plus_single_point(R, "scc", scc_lam=lam_h2plus)
        e_fs, _ = _h2plus_single_point(R, "both", alpha=alpha_h2plus, scc_lam=lam_h2plus)

        dE_F = (e_f - e_pbe) * 1000
        dE_S = (e_s - e_pbe) * 1000
        dE_FS = (e_fs - e_pbe) * 1000
        eps = dE_FS - (dE_F + dE_S)
        denom = abs(dE_F) + abs(dE_S)
        eps_pct = 100 * eps / denom if denom > 0.1 else 0

        print(f"{R:6.3f}  {dE_F:+10.2f}  {dE_S:+10.2f}  {dE_FS:+10.2f}  {eps:+10.2f}  {eps_pct:+8.1f}%")

    # H₂
    print(f"\n--- H₂ (α={alpha_h2}, λ={lam_h2}) ---")
    distances_h2 = np.array([0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0])

    print(f"{'R(Å)':>6}  {'ΔE_F':>10}  {'ΔE_S':>10}  {'ΔE_FS':>10}  {'ε(R)':>10}  {'ε%':>8}")
    print("-" * 65)

    for R in distances_h2:
        e_pbe, _ = _h2_single_point(R, "pbe")
        e_f, _ = _h2_single_point(R, "fisher", alpha=alpha_h2)
        e_s, _ = _h2_single_point(R, "scc", scc_lam=lam_h2)
        e_fs, _ = _h2_single_point(R, "both", alpha=alpha_h2, scc_lam=lam_h2)

        dE_F = (e_f - e_pbe) * 1000
        dE_S = (e_s - e_pbe) * 1000
        dE_FS = (e_fs - e_pbe) * 1000
        eps = dE_FS - (dE_F + dE_S)
        denom = abs(dE_F) + abs(dE_S)
        eps_pct = 100 * eps / denom if denom > 0.1 else 0

        print(f"{R:6.3f}  {dE_F:+10.2f}  {dE_S:+10.2f}  {dE_FS:+10.2f}  {eps:+10.2f}  {eps_pct:+8.1f}%")


# =============================================================================
# Baseline Comparison (Water)
# =============================================================================

def run_baseline_comparison():
    """Compare PBE vs PBE+Fisher vs PBE+entropy on water."""
    mol = gto.M(
        atom="""
        O 0.000000  0.000000  0.000000
        H 0.000000 -0.757000  0.587000
        H 0.000000  0.757000  0.587000
        """,
        basis="def2-svp",
        verbose=0,
    )

    alpha_vals = [0.0, 1e-3, 5e-3, 1e-2, 2e-2]

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON: Water / def2-SVP")
    print("=" * 80)

    # PBE reference
    print("\n--- PBE Reference ---")
    mf_pbe = dft.RKS(mol)
    mf_pbe.xc = 'PBE'
    mf_pbe.grids.level = 4
    mf_pbe.kernel()
    e_pbe = mf_pbe.e_tot
    homo, lumo, gap = get_orbital_energies(mf_pbe)
    print(f"E_tot = {e_pbe:.10f}  HOMO = {homo:.6f}  LUMO = {lumo:.6f}  gap = {gap*27.2114:.4f} eV")

    # Fisher scan
    print("\n--- PBE + Fisher (α scan) ---")
    print(f"{'α':>8}  {'E_tot':>14}  {'ΔE(mHa)':>10}  {'E_fisher':>12}  {'gap(eV)':>8}")
    print("-" * 60)

    dm = None
    for a in alpha_vals:
        mf = dft.RKS(mol)
        mf.grids.level = 4
        mf.define_xc_(make_rift_xc(alpha=a), xctype="GGA")
        if dm is not None:
            mf.kernel(dm0=dm)
        else:
            mf.kernel()
        dm = mf.make_rdm1()

        e_fisher, _ = compute_rift_energy(mf, a, 0.0)
        homo, lumo, gap = get_orbital_energies(mf)
        de = (mf.e_tot - e_pbe) * 1000
        print(f"{a:8.3g}  {mf.e_tot:14.8f}  {de:+10.4f}  {e_fisher:12.6f}  {gap*27.2114:8.4f}")

    return mf_pbe


# =============================================================================
# Harmonizer Comparison
# =============================================================================

def run_harmonizer_comparison():
    """Compare different gate combinations."""
    print("\n" + "=" * 80)
    print("HARMONIZER COMPARISON: f(s)*g(ρ) combined gate")
    print("=" * 80)

    mol = gto.M(
        atom="""
        O 0.000000  0.000000  0.000000
        H 0.000000 -0.757000  0.587000
        H 0.000000  0.757000  0.587000
        """,
        basis="def2-svp",
        verbose=0,
    )

    # PBE reference
    mf_pbe = dft.RKS(mol)
    mf_pbe.xc = 'PBE'
    mf_pbe.grids.level = 4
    mf_pbe.kernel()
    e_pbe = mf_pbe.e_tot
    print(f"\nPBE Reference: E = {e_pbe:.10f}")

    alpha = 0.02
    rho_c = 1.0

    variants = [
        ("None (raw)", False, 0.0),
        ("f(s) only", True, 0.0),
        ("g(ρ) only", False, rho_c),
        ("f(s)*g(ρ)", True, rho_c),
    ]

    print(f"\n--- Gate Comparison at α={alpha} ---")
    print(f"{'Variant':>20}  {'E_tot':>14}  {'ΔE(mHa)':>10}  {'E_fisher':>10}")
    print("-" * 60)

    for name, harm, rc in variants:
        mf = dft.RKS(mol)
        mf.grids.level = 4
        mf.define_xc_(make_rift_xc(alpha=alpha, harmonize=harm, rho_core=rc), xctype="GGA")
        mf.kernel()
        e_fish, _ = compute_rift_energy(mf, alpha, 0.0, harmonize=harm, rho_core=rc)
        de = (mf.e_tot - e_pbe) * 1000
        print(f"{name:>20}  {mf.e_tot:14.8f}  {de:+10.2f}  {e_fish:10.4f}")


# =============================================================================
# Density Diagnostics
# =============================================================================

def run_density_diagnostics():
    """Compute density difference diagnostics for RIFT vs PBE."""
    print("\n" + "=" * 80)
    print("DENSITY DIAGNOSTICS: ρ(RIFT) - ρ(PBE)")
    print("=" * 80)

    mol = gto.M(
        atom="""
        O 0.000000  0.000000  0.000000
        H 0.000000 -0.757000  0.587000
        H 0.000000  0.757000  0.587000
        """,
        basis="def2-svp",
        verbose=0,
    )

    mf_pbe = dft.RKS(mol)
    mf_pbe.xc = 'PBE'
    mf_pbe.grids.level = 4
    mf_pbe.kernel()

    alpha_vals = [1e-3, 5e-3, 1e-2, 2e-2]

    print(f"\n{'α':>8}  {'L¹':>12}  {'L²':>12}  {'weighted':>12}  {'max|Δρ|':>12}")
    print("-" * 65)

    dm = mf_pbe.make_rdm1()
    for a in alpha_vals:
        mf = dft.RKS(mol)
        mf.grids.level = 4
        mf.define_xc_(make_rift_xc(alpha=a), xctype="GGA")
        mf.kernel(dm0=dm)
        dm = mf.make_rdm1()

        diag = compute_density_diagnostics(mf_pbe, mf)
        print(f"{a:8.3g}  {diag['L1']:12.6e}  {diag['L2']:12.6e}  "
              f"{diag['weighted']:12.6e}  {abs(diag['max_delta']):12.6e}")


# =============================================================================
# ECP Comparison
# =============================================================================

def run_ecp_comparison():
    """Compare Fisher term magnitude with all-electron vs ECP."""
    print("\n" + "=" * 80)
    print("ECP CORE-SUPPRESSION TEST: Water")
    print("=" * 80)

    mol_ae = gto.M(
        atom="""
        O 0.000000  0.000000  0.000000
        H 0.000000 -0.757000  0.587000
        H 0.000000  0.757000  0.587000
        """,
        basis="def2-svp",
        verbose=0,
    )

    mol_ecp = gto.M(
        atom="""
        O 0.000000  0.000000  0.000000
        H 0.000000 -0.757000  0.587000
        H 0.000000  0.757000  0.587000
        """,
        basis="def2-svp",
        ecp="def2-svp",
        verbose=0,
    )

    alpha_vals = [0.0, 1e-3, 5e-3, 1e-2, 2e-2]

    print("\n--- All-Electron ---")
    print(f"{'α':>8}  {'E_tot':>14}  {'E_fisher':>12}")
    print("-" * 40)

    dm = None
    for a in alpha_vals:
        mf = dft.RKS(mol_ae)
        mf.grids.level = 4
        mf.define_xc_(make_rift_xc(alpha=a), xctype="GGA")
        if dm is not None:
            mf.kernel(dm0=dm)
        else:
            mf.kernel()
        dm = mf.make_rdm1()
        e_fisher, _ = compute_rift_energy(mf, a, 0.0)
        print(f"{a:8.3g}  {mf.e_tot:14.8f}  {e_fisher:12.6f}")

    print("\n--- With ECP ---")
    print(f"{'α':>8}  {'E_tot':>14}  {'E_fisher':>12}")
    print("-" * 40)

    dm = None
    for a in alpha_vals:
        mf = dft.RKS(mol_ecp)
        mf.grids.level = 4
        mf.define_xc_(make_rift_xc(alpha=a), xctype="GGA")
        if dm is not None:
            mf.kernel(dm0=dm)
        else:
            mf.kernel()
        dm = mf.make_rdm1()
        e_fisher, _ = compute_rift_energy(mf, a, 0.0)
        print(f"{a:8.3g}  {mf.e_tot:14.8f}  {e_fisher:12.6f}")


# =============================================================================
# SCC Gate Diagnostics
# =============================================================================

def compute_scc_gate_diagnostics(mf, rho_s=0.1, h_power=6):
    """
    Compute SCC v2 gate diagnostics for a converged calculation.

    Returns dict with:
        w_avg: density-weighted average of iso-orbital gate w(z)
        w_eff_avg: density-weighted average of effective gate w_eff = w(z) × h(ρ)
        z_avg: density-weighted average of iso-orbital indicator z
        h_avg: density-weighted average of density suppression h(ρ)
        frac_w_high: fraction of density with w > 0.9
        frac_weff_high: fraction of density with w_eff > 0.1
        nel: total electron count
    """
    from pyscf.dft.numint import NumInt

    ni = mf._numint if hasattr(mf, "_numint") else NumInt()
    mol = mf.mol
    grids = mf.grids
    dm = mf.make_rdm1()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ

    if grids.coords is None:
        grids.build()

    ao = ni.eval_ao(mol, grids.coords, deriv=1)
    ao0, aox, aoy, aoz = ao[0], ao[1], ao[2], ao[3]

    dm_arr = np.asarray(dm)
    is_uks = dm_arr.ndim == 3 and dm_arr.shape[0] == 2

    if is_uks:
        rho_a = ni.eval_rho(mol, ao, dm_arr[0], xctype="GGA", hermi=1)
        rho_b = ni.eval_rho(mol, ao, dm_arr[1], xctype="GGA", hermi=1)
        rho_tot = rho_a[0] + rho_b[0]
        grad_tot = rho_a[1:4] + rho_b[1:4]
    else:
        rho_full = ni.eval_rho(mol, ao, dm_arr, xctype="GGA", hermi=1)
        rho_tot = rho_full[0]
        grad_tot = rho_full[1:4]

    # Compute tau
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

    # z = tauW / tau
    sigma_tot = np.einsum("ig,ig->g", grad_tot, grad_tot)
    rho_eff = np.maximum(rho_tot, 1e-12)
    tauW = sigma_tot / (8.0 * rho_eff)
    tau_eff = np.maximum(tau_tot, 1e-12)
    z = np.clip(tauW / tau_eff, 0.0, 1.0)

    # Iso-orbital gate w(z)
    w_power = 4
    zn = z ** w_power
    on = (1.0 - z) ** w_power
    w = zn / (zn + on + 1e-30)

    # Density suppression gate h(ρ)
    h = 1.0 / (1.0 + (rho_eff / rho_s) ** h_power)

    # Effective gate
    w_eff = w * h

    # Compute diagnostics
    wt = grids.weights
    rho_weighted = rho_tot * wt
    total_charge = np.sum(rho_weighted)

    w_avg = np.sum(w * rho_weighted) / np.maximum(total_charge, 1e-30)
    w_eff_avg = np.sum(w_eff * rho_weighted) / np.maximum(total_charge, 1e-30)
    z_avg = np.sum(z * rho_weighted) / np.maximum(total_charge, 1e-30)
    h_avg = np.sum(h * rho_weighted) / np.maximum(total_charge, 1e-30)

    # Fraction of charge with w > 0.9 (old metric)
    high_w_mask = w > 0.9
    frac_w_high = np.sum(rho_weighted[high_w_mask]) / np.maximum(total_charge, 1e-30)

    # Fraction of charge with w_eff > 0.1 (new metric - where SCC is active)
    high_weff_mask = w_eff > 0.1
    frac_weff_high = np.sum(rho_weighted[high_weff_mask]) / np.maximum(total_charge, 1e-30)

    return {
        'w_avg': w_avg,
        'w_eff_avg': w_eff_avg,
        'z_avg': z_avg,
        'h_avg': h_avg,
        'frac_w_high': frac_w_high,
        'frac_weff_high': frac_weff_high,
        'nel': total_charge,
    }


# =============================================================================
# CO / NO Single-Point Helpers
# =============================================================================

def _co_single_point(R, mode, alpha=0.0, scc_lam=0.1):
    """
    Run single CO calculation.

    Modes: pbe, hf, pbe0, fisher, scc, both
    """
    mol = gto.M(
        atom=f"C 0 0 0; O 0 0 {R}",
        basis="def2-svp",
        charge=0,
        spin=0,
        verbose=0,
    )

    if mode == "hf":
        mf = scf.RHF(mol)
        mf.max_cycle = 150
        mf.kernel()
        return mf.e_tot, mf.converged, None

    mf = dft.RKS(mol)
    mf.grids.level = 4
    mf.max_cycle = 150

    if mode == "pbe":
        mf.xc = BASE_XC
    elif mode == "pbe0":
        mf.xc = "PBE0"
    elif mode == "fisher":
        FisherModule(FisherCfg(alpha=alpha)).attach(mf)
    elif mode == "scc":
        mf.xc = BASE_XC
        mf.kernel()
        if mf.converged:
            dm0 = mf.make_rdm1()
            SCCModule(SCCCfg(lam=scc_lam)).attach(mf)
            mf.kernel(dm0=dm0)
        diag = compute_scc_gate_diagnostics(mf)
        return mf.e_tot, mf.converged, diag
    elif mode == "both":
        FisherModule(FisherCfg(alpha=alpha)).attach(mf)
        mf.kernel()
        if mf.converged:
            dm0 = mf.make_rdm1()
            SCCModule(SCCCfg(lam=scc_lam)).attach(mf)
            mf.kernel(dm0=dm0)
        diag = compute_scc_gate_diagnostics(mf)
        return mf.e_tot, mf.converged, diag
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mf.kernel()
    diag = compute_scc_gate_diagnostics(mf) if mode in ("pbe", "pbe0", "fisher") else None
    return mf.e_tot, mf.converged, diag


def _no_single_point(R, mode, alpha=0.0, scc_lam=0.1):
    """
    Run single NO calculation (open-shell doublet).

    Modes: pbe, uhf, pbe0, fisher, scc, both
    """
    mol = gto.M(
        atom=f"N 0 0 0; O 0 0 {R}",
        basis="def2-svp",
        charge=0,
        spin=1,  # Doublet
        verbose=0,
    )

    if mode == "uhf":
        mf = scf.UHF(mol)
        mf.max_cycle = 150
        mf.kernel()
        return mf.e_tot, mf.converged, None

    mf = dft.UKS(mol)
    mf.grids.level = 4
    mf.max_cycle = 150

    if mode == "pbe":
        mf.xc = BASE_XC
    elif mode == "pbe0":
        mf.xc = "PBE0"
    elif mode == "fisher":
        FisherModule(FisherCfg(alpha=alpha)).attach(mf)
    elif mode == "scc":
        mf.xc = BASE_XC
        mf.kernel()
        if mf.converged:
            dm0 = mf.make_rdm1()
            SCCModule(SCCCfg(lam=scc_lam)).attach(mf)
            mf.kernel(dm0=dm0)
        diag = compute_scc_gate_diagnostics(mf)
        return mf.e_tot, mf.converged, diag
    elif mode == "both":
        FisherModule(FisherCfg(alpha=alpha)).attach(mf)
        mf.kernel()
        if mf.converged:
            dm0 = mf.make_rdm1()
            SCCModule(SCCCfg(lam=scc_lam)).attach(mf)
            mf.kernel(dm0=dm0)
        diag = compute_scc_gate_diagnostics(mf)
        return mf.e_tot, mf.converged, diag
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mf.kernel()
    diag = compute_scc_gate_diagnostics(mf) if mode in ("pbe", "pbe0", "fisher") else None
    return mf.e_tot, mf.converged, diag


# =============================================================================
# CO / NO Dissociation Tests
# =============================================================================

def run_co_test(alpha=0.005, scc_lam=0.1):
    """
    CO dissociation: test Fisher + SCC on a real molecule.

    CO has:
    - Strong triple bond (overbinding tendency in PBE)
    - Known bond length errors
    - 14 electrons (many-electron test for SCC gate)
    """
    print("\n" + "=" * 80)
    print("CO DISSOCIATION TEST")
    print("=" * 80)
    print(f"Parameters: α = {alpha}, λ = {scc_lam}")
    print("Testing Fisher + SCC on closed-shell molecule with strong bond.\n")

    # Experimental: R_e = 1.128 Å
    R_exp = 1.128
    distances = np.array([0.9, 1.0, 1.128, 1.2, 1.4, 1.6, 2.0, 2.5])

    modes = ["pbe", "fisher", "scc", "both"]

    print(f"{'R(Å)':>6}  {'PBE':>12}  {'Fisher':>12}  {'SCC':>12}  {'Both':>12}  {'⟨w⟩':>6}  {'⟨w_eff⟩':>8}  conv")
    print("-" * 90)

    results = {m: [] for m in modes}

    for R in distances:
        row = f"{R:6.3f}"
        conv = ""
        w_avg = None
        w_eff_avg = None

        for mode in modes:
            e, c, diag = _co_single_point(R, mode, alpha=alpha, scc_lam=scc_lam)
            results[mode].append({'R': R, 'E': e, 'conv': c, 'diag': diag})
            row += f"  {e:12.8f}"
            conv += "✓" if c else "✗"
            if mode == "scc" and diag:
                w_avg = diag['w_avg']
                w_eff_avg = diag.get('w_eff_avg', w_avg)

        if w_avg is not None:
            row += f"  {w_avg:6.3f}  {w_eff_avg:8.4f}"
        else:
            row += f"  {'N/A':>6}  {'N/A':>8}"

        print(row + f"  {conv}")

    # Binding energies relative to large R
    print(f"\n--- Binding Energy (E - E_∞) in mHa ---")
    print(f"{'R(Å)':>6}  {'PBE':>10}  {'Fisher':>10}  {'SCC':>10}  {'Both':>10}  {'Δ_F':>10}  {'Δ_S':>10}  {'Δ_FS':>10}")
    print("-" * 95)

    e_inf = {m: results[m][-1]['E'] for m in modes}

    for i, R in enumerate(distances):
        be = {m: (results[m][i]['E'] - e_inf[m]) * 1000 for m in modes}
        dF = be['fisher'] - be['pbe']
        dS = be['scc'] - be['pbe']
        dFS = be['both'] - be['pbe']
        print(f"{R:6.3f}  {be['pbe']:+10.2f}  {be['fisher']:+10.2f}  {be['scc']:+10.2f}  {be['both']:+10.2f}  {dF:+10.2f}  {dS:+10.2f}  {dFS:+10.2f}")

    # Find equilibrium
    idx_min = {m: np.argmin([r['E'] for r in results[m]]) for m in modes}
    print(f"\n--- Equilibrium Bond Length ---")
    print(f"  Experimental: {R_exp:.3f} Å")
    for m in modes:
        print(f"  {m:>10}: {distances[idx_min[m]]:.3f} Å")

    # Gate diagnostics at equilibrium
    print(f"\n--- SCC v2 Gate Diagnostics at R ≈ {R_exp} Å ---")
    idx_eq = np.argmin(np.abs(distances - R_exp))
    for m in ["pbe", "scc", "both"]:
        diag = results[m][idx_eq].get('diag')
        if diag:
            w_eff = diag.get('w_eff_avg', diag['w_avg'])
            h = diag.get('h_avg', 1.0)
            print(f"  {m:>10}: ⟨w⟩ = {diag['w_avg']:.3f}, ⟨h⟩ = {h:.3f}, ⟨w_eff⟩ = {w_eff:.4f}, ⟨z⟩ = {diag['z_avg']:.3f}")

    return results


def run_no_test(alpha=0.005, scc_lam=0.1):
    """
    NO dissociation: test Fisher + SCC on open-shell molecule.

    NO has:
    - Open-shell doublet (tests UKS stability)
    - Partial radical character
    - 15 electrons
    """
    print("\n" + "=" * 80)
    print("NO DISSOCIATION TEST")
    print("=" * 80)
    print(f"Parameters: α = {alpha}, λ = {scc_lam}")
    print("Testing Fisher + SCC on open-shell doublet.\n")

    # Experimental: R_e = 1.151 Å
    R_exp = 1.151
    distances = np.array([0.9, 1.0, 1.151, 1.2, 1.4, 1.6, 2.0, 2.5])

    modes = ["pbe", "fisher", "scc", "both"]

    print(f"{'R(Å)':>6}  {'PBE':>12}  {'Fisher':>12}  {'SCC':>12}  {'Both':>12}  {'⟨w⟩':>6}  {'⟨w_eff⟩':>8}  conv")
    print("-" * 90)

    results = {m: [] for m in modes}

    for R in distances:
        row = f"{R:6.3f}"
        conv = ""
        w_avg = None
        w_eff_avg = None

        for mode in modes:
            e, c, diag = _no_single_point(R, mode, alpha=alpha, scc_lam=scc_lam)
            results[mode].append({'R': R, 'E': e, 'conv': c, 'diag': diag})
            row += f"  {e:12.8f}"
            conv += "✓" if c else "✗"
            if mode == "scc" and diag:
                w_avg = diag['w_avg']
                w_eff_avg = diag.get('w_eff_avg', w_avg)

        if w_avg is not None:
            row += f"  {w_avg:6.3f}  {w_eff_avg:8.4f}"
        else:
            row += f"  {'N/A':>6}  {'N/A':>8}"

        print(row + f"  {conv}")

    # Binding energies
    print(f"\n--- Binding Energy (E - E_∞) in mHa ---")
    print(f"{'R(Å)':>6}  {'PBE':>10}  {'Fisher':>10}  {'SCC':>10}  {'Both':>10}  {'Δ_F':>10}  {'Δ_S':>10}  {'Δ_FS':>10}")
    print("-" * 95)

    e_inf = {m: results[m][-1]['E'] for m in modes}

    for i, R in enumerate(distances):
        be = {m: (results[m][i]['E'] - e_inf[m]) * 1000 for m in modes}
        dF = be['fisher'] - be['pbe']
        dS = be['scc'] - be['pbe']
        dFS = be['both'] - be['pbe']
        print(f"{R:6.3f}  {be['pbe']:+10.2f}  {be['fisher']:+10.2f}  {be['scc']:+10.2f}  {be['both']:+10.2f}  {dF:+10.2f}  {dS:+10.2f}  {dFS:+10.2f}")

    # Find equilibrium
    idx_min = {m: np.argmin([r['E'] for r in results[m]]) for m in modes}
    print(f"\n--- Equilibrium Bond Length ---")
    print(f"  Experimental: {R_exp:.3f} Å")
    for m in modes:
        print(f"  {m:>10}: {distances[idx_min[m]]:.3f} Å")

    # Gate diagnostics
    print(f"\n--- SCC v2 Gate Diagnostics at R ≈ {R_exp} Å ---")
    idx_eq = np.argmin(np.abs(distances - R_exp))
    for m in ["pbe", "scc", "both"]:
        diag = results[m][idx_eq].get('diag')
        if diag:
            w_eff = diag.get('w_eff_avg', diag['w_avg'])
            h = diag.get('h_avg', 1.0)
            print(f"  {m:>10}: ⟨w⟩ = {diag['w_avg']:.3f}, ⟨h⟩ = {h:.3f}, ⟨w_eff⟩ = {w_eff:.4f}, ⟨z⟩ = {diag['z_avg']:.3f}")

    return results


def run_co_no_combined():
    """Run both CO and NO tests with default parameters."""
    run_co_test(alpha=0.005, scc_lam=0.1)
    run_no_test(alpha=0.005, scc_lam=0.1)


def run_rho_s_sweep(rho_s_values=None, lam=0.1):
    """
    Sweep ρ_s to find optimal density suppression threshold.

    For each ρ_s, reports:
      - H₂⁺ at R=2.0Å: ⟨w_eff⟩, ΔE(SCC-PBE)
      - CO at R=1.128Å: ⟨w_eff⟩, ΔBE(SCC-PBE)
      - NO at R=1.151Å: ⟨w_eff⟩, ΔBE(SCC-PBE)
    """
    if rho_s_values is None:
        rho_s_values = [0.10, 0.08, 0.05, 0.03]

    print("\n" + "=" * 80)
    print(f"ρ_s SWEEP: Finding optimal density suppression threshold")
    print(f"Fixed: λ = {lam}, h_power = 6, w_power = 4")
    print("=" * 80)

    # Test points
    R_h2plus = 2.0
    R_co = 1.128
    R_no = 1.151
    R_inf = 5.0  # For binding energy reference

    results = []

    for rho_s in rho_s_values:
        print(f"\n--- ρ_s = {rho_s} ---")
        row = {'rho_s': rho_s}

        # H₂⁺ at R=2.0
        mol_h2plus = gto.M(
            atom=f"H 0 0 0; H 0 0 {R_h2plus}",
            basis="def2-svp", charge=1, spin=1, verbose=0,
        )
        mol_h2plus_inf = gto.M(
            atom=f"H 0 0 0; H 0 0 {R_inf}",
            basis="def2-svp", charge=1, spin=1, verbose=0,
        )

        # PBE reference
        mf_pbe = dft.UKS(mol_h2plus)
        mf_pbe.xc = BASE_XC
        mf_pbe.grids.level = 4
        mf_pbe.kernel()
        e_pbe = mf_pbe.e_tot

        mf_pbe_inf = dft.UKS(mol_h2plus_inf)
        mf_pbe_inf.xc = BASE_XC
        mf_pbe_inf.grids.level = 4
        mf_pbe_inf.kernel()
        e_pbe_inf = mf_pbe_inf.e_tot

        # SCC with this rho_s
        mf_scc = dft.UKS(mol_h2plus)
        mf_scc.xc = BASE_XC
        mf_scc.grids.level = 4
        mf_scc.kernel()
        dm0 = mf_scc.make_rdm1()
        SCCModule(SCCCfg(lam=lam, rho_s=rho_s)).attach(mf_scc)
        mf_scc.kernel(dm0=dm0)

        mf_scc_inf = dft.UKS(mol_h2plus_inf)
        mf_scc_inf.xc = BASE_XC
        mf_scc_inf.grids.level = 4
        mf_scc_inf.kernel()
        dm0_inf = mf_scc_inf.make_rdm1()
        SCCModule(SCCCfg(lam=lam, rho_s=rho_s)).attach(mf_scc_inf)
        mf_scc_inf.kernel(dm0=dm0_inf)

        diag_h2plus = compute_scc_gate_diagnostics(mf_scc, rho_s=rho_s)
        be_pbe = (e_pbe - e_pbe_inf) * 1000
        be_scc = (mf_scc.e_tot - mf_scc_inf.e_tot) * 1000
        delta_be_h2plus = be_scc - be_pbe

        row['h2plus_w_eff'] = diag_h2plus['w_eff_avg']
        row['h2plus_delta_be'] = delta_be_h2plus
        print(f"  H₂⁺ (R=2.0): ⟨w_eff⟩ = {diag_h2plus['w_eff_avg']:.4f}, ΔBE = {delta_be_h2plus:+.2f} mHa")

        # CO at equilibrium
        mol_co = gto.M(atom=f"C 0 0 0; O 0 0 {R_co}", basis="def2-svp", verbose=0)
        mol_co_inf = gto.M(atom=f"C 0 0 0; O 0 0 {R_inf}", basis="def2-svp", verbose=0)

        mf_pbe_co = dft.RKS(mol_co)
        mf_pbe_co.xc = BASE_XC
        mf_pbe_co.grids.level = 4
        mf_pbe_co.kernel()

        mf_pbe_co_inf = dft.RKS(mol_co_inf)
        mf_pbe_co_inf.xc = BASE_XC
        mf_pbe_co_inf.grids.level = 4
        mf_pbe_co_inf.kernel()

        mf_scc_co = dft.RKS(mol_co)
        mf_scc_co.xc = BASE_XC
        mf_scc_co.grids.level = 4
        mf_scc_co.kernel()
        dm0_co = mf_scc_co.make_rdm1()
        SCCModule(SCCCfg(lam=lam, rho_s=rho_s)).attach(mf_scc_co)
        mf_scc_co.kernel(dm0=dm0_co)

        mf_scc_co_inf = dft.RKS(mol_co_inf)
        mf_scc_co_inf.xc = BASE_XC
        mf_scc_co_inf.grids.level = 4
        mf_scc_co_inf.kernel()
        dm0_co_inf = mf_scc_co_inf.make_rdm1()
        SCCModule(SCCCfg(lam=lam, rho_s=rho_s)).attach(mf_scc_co_inf)
        mf_scc_co_inf.kernel(dm0=dm0_co_inf)

        diag_co = compute_scc_gate_diagnostics(mf_scc_co, rho_s=rho_s)
        be_pbe_co = (mf_pbe_co.e_tot - mf_pbe_co_inf.e_tot) * 1000
        be_scc_co = (mf_scc_co.e_tot - mf_scc_co_inf.e_tot) * 1000
        delta_be_co = be_scc_co - be_pbe_co

        row['co_w_eff'] = diag_co['w_eff_avg']
        row['co_delta_be'] = delta_be_co
        print(f"  CO (R=1.128): ⟨w_eff⟩ = {diag_co['w_eff_avg']:.4f}, ΔBE = {delta_be_co:+.2f} mHa")

        # NO at equilibrium
        mol_no = gto.M(atom=f"N 0 0 0; O 0 0 {R_no}", basis="def2-svp", spin=1, verbose=0)
        mol_no_inf = gto.M(atom=f"N 0 0 0; O 0 0 {R_inf}", basis="def2-svp", spin=1, verbose=0)

        mf_pbe_no = dft.UKS(mol_no)
        mf_pbe_no.xc = BASE_XC
        mf_pbe_no.grids.level = 4
        mf_pbe_no.kernel()

        mf_pbe_no_inf = dft.UKS(mol_no_inf)
        mf_pbe_no_inf.xc = BASE_XC
        mf_pbe_no_inf.grids.level = 4
        mf_pbe_no_inf.kernel()

        mf_scc_no = dft.UKS(mol_no)
        mf_scc_no.xc = BASE_XC
        mf_scc_no.grids.level = 4
        mf_scc_no.kernel()
        dm0_no = mf_scc_no.make_rdm1()
        SCCModule(SCCCfg(lam=lam, rho_s=rho_s)).attach(mf_scc_no)
        mf_scc_no.kernel(dm0=dm0_no)

        mf_scc_no_inf = dft.UKS(mol_no_inf)
        mf_scc_no_inf.xc = BASE_XC
        mf_scc_no_inf.grids.level = 4
        mf_scc_no_inf.kernel()
        dm0_no_inf = mf_scc_no_inf.make_rdm1()
        SCCModule(SCCCfg(lam=lam, rho_s=rho_s)).attach(mf_scc_no_inf)
        mf_scc_no_inf.kernel(dm0=dm0_no_inf)

        diag_no = compute_scc_gate_diagnostics(mf_scc_no, rho_s=rho_s)
        be_pbe_no = (mf_pbe_no.e_tot - mf_pbe_no_inf.e_tot) * 1000
        be_scc_no = (mf_scc_no.e_tot - mf_scc_no_inf.e_tot) * 1000
        delta_be_no = be_scc_no - be_pbe_no

        row['no_w_eff'] = diag_no['w_eff_avg']
        row['no_delta_be'] = delta_be_no
        print(f"  NO (R=1.151): ⟨w_eff⟩ = {diag_no['w_eff_avg']:.4f}, ΔBE = {delta_be_no:+.2f} mHa")

        results.append(row)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'ρ_s':>6}  {'H₂⁺ ⟨w_eff⟩':>12}  {'H₂⁺ ΔBE':>10}  {'CO ⟨w_eff⟩':>11}  {'CO ΔBE':>10}  {'NO ⟨w_eff⟩':>11}  {'NO ΔBE':>10}")
    print("-" * 85)
    for r in results:
        print(f"{r['rho_s']:6.2f}  {r['h2plus_w_eff']:12.4f}  {r['h2plus_delta_be']:+10.2f}  "
              f"{r['co_w_eff']:11.4f}  {r['co_delta_be']:+10.2f}  "
              f"{r['no_w_eff']:11.4f}  {r['no_delta_be']:+10.2f}")

    print("\nTargets:")
    print("  H₂⁺: ΔBE should be negative (SIE correction), ideally -20 to -40 mHa")
    print("  CO/NO: |ΔBE| < 50 mHa (minimal perturbation)")

    return results
