#!/usr/bin/env python
"""Tight ρ_s sweep for CO/NO - focused test."""
import numpy as np
from pyscf import gto, dft
from rift_functionals import BASE_XC, SCCCfg, SCCModule
from experiments import compute_scc_gate_diagnostics

def run_co_scan(rho_s, lam=0.1):
    """CO dissociation scan at given ρ_s."""
    print(f"\n=== CO at ρ_s={rho_s} ===")

    distances = np.array([0.9, 1.0, 1.128, 1.2, 1.4, 1.6, 2.0])
    R_exp = 1.128

    results_pbe = []
    results_scc = []

    for R in distances:
        mol = gto.M(atom=f"C 0 0 0; O 0 0 {R}", basis="def2-svp", verbose=0)

        # PBE
        mf_pbe = dft.RKS(mol)
        mf_pbe.xc = BASE_XC
        mf_pbe.grids.level = 4
        mf_pbe.kernel()
        results_pbe.append(mf_pbe.e_tot)

        # SCC
        mf_scc = dft.RKS(mol)
        mf_scc.xc = BASE_XC
        mf_scc.grids.level = 4
        mf_scc.kernel()
        dm0 = mf_scc.make_rdm1()
        SCCModule(SCCCfg(lam=lam, rho_s=rho_s)).attach(mf_scc)
        mf_scc.kernel(dm0=dm0)
        results_scc.append(mf_scc.e_tot)

        if abs(R - R_exp) < 0.01:
            diag = compute_scc_gate_diagnostics(mf_scc, rho_s=rho_s)

    # Find equilibria
    idx_pbe = np.argmin(results_pbe)
    idx_scc = np.argmin(results_scc)
    R_eq_pbe = distances[idx_pbe]
    R_eq_scc = distances[idx_scc]

    # Binding energies (relative to R=2.0)
    e_inf_pbe = results_pbe[-1]
    e_inf_scc = results_scc[-1]

    be_pbe = [(e - e_inf_pbe) * 1000 for e in results_pbe]
    be_scc = [(e - e_inf_scc) * 1000 for e in results_scc]

    # At experimental equilibrium
    idx_eq = np.argmin(np.abs(distances - R_exp))
    delta_be = be_scc[idx_eq] - be_pbe[idx_eq]

    print(f"  R_eq(PBE) = {R_eq_pbe:.3f} Å, R_eq(SCC) = {R_eq_scc:.3f} Å")
    print(f"  ΔR_eq = {R_eq_scc - R_eq_pbe:+.3f} Å")
    print(f"  ΔBE(eq) = {delta_be:+.1f} mHa")
    print(f"  ⟨w_eff⟩ = {diag['w_eff_avg']:.4f}")
    print(f"  frac(w_eff>0.1) = {diag.get('frac_weff_high', 0):.1%}")

    return {'rho_s': rho_s, 'delta_be': delta_be, 'delta_r': R_eq_scc - R_eq_pbe,
            'w_eff': diag['w_eff_avg'], 'system': 'CO'}


def run_no_scan(rho_s, lam=0.1):
    """NO dissociation scan at given ρ_s."""
    print(f"\n=== NO at ρ_s={rho_s} ===")

    distances = np.array([0.9, 1.0, 1.151, 1.2, 1.4, 1.6, 2.0])
    R_exp = 1.151

    results_pbe = []
    results_scc = []

    for R in distances:
        mol = gto.M(atom=f"N 0 0 0; O 0 0 {R}", basis="def2-svp", spin=1, verbose=0)

        # PBE
        mf_pbe = dft.UKS(mol)
        mf_pbe.xc = BASE_XC
        mf_pbe.grids.level = 4
        mf_pbe.kernel()
        results_pbe.append(mf_pbe.e_tot)

        # SCC
        mf_scc = dft.UKS(mol)
        mf_scc.xc = BASE_XC
        mf_scc.grids.level = 4
        mf_scc.kernel()
        dm0 = mf_scc.make_rdm1()
        SCCModule(SCCCfg(lam=lam, rho_s=rho_s)).attach(mf_scc)
        mf_scc.kernel(dm0=dm0)
        results_scc.append(mf_scc.e_tot)

        if abs(R - R_exp) < 0.01:
            diag = compute_scc_gate_diagnostics(mf_scc, rho_s=rho_s)

    # Find equilibria
    idx_pbe = np.argmin(results_pbe)
    idx_scc = np.argmin(results_scc)
    R_eq_pbe = distances[idx_pbe]
    R_eq_scc = distances[idx_scc]

    # Binding energies
    e_inf_pbe = results_pbe[-1]
    e_inf_scc = results_scc[-1]

    be_pbe = [(e - e_inf_pbe) * 1000 for e in results_pbe]
    be_scc = [(e - e_inf_scc) * 1000 for e in results_scc]

    idx_eq = np.argmin(np.abs(distances - R_exp))
    delta_be = be_scc[idx_eq] - be_pbe[idx_eq]

    print(f"  R_eq(PBE) = {R_eq_pbe:.3f} Å, R_eq(SCC) = {R_eq_scc:.3f} Å")
    print(f"  ΔR_eq = {R_eq_scc - R_eq_pbe:+.3f} Å")
    print(f"  ΔBE(eq) = {delta_be:+.1f} mHa")
    print(f"  ⟨w_eff⟩ = {diag['w_eff_avg']:.4f}")
    print(f"  frac(w_eff>0.1) = {diag.get('frac_weff_high', 0):.1%}")

    return {'rho_s': rho_s, 'delta_be': delta_be, 'delta_r': R_eq_scc - R_eq_pbe,
            'w_eff': diag['w_eff_avg'], 'system': 'NO'}


if __name__ == "__main__":
    print("=" * 60)
    print("TIGHT ρ_s SWEEP: CO/NO only")
    print("Fixed: λ=0.1, h_power=6, w_power=4")
    print("=" * 60)

    results = []

    for rho_s in [0.05, 0.03]:
        results.append(run_co_scan(rho_s))
        results.append(run_no_scan(rho_s))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'System':>6} {'ρ_s':>6} {'ΔBE(mHa)':>10} {'ΔR_eq':>8} {'⟨w_eff⟩':>8}")
    print("-" * 45)
    for r in results:
        print(f"{r['system']:>6} {r['rho_s']:6.2f} {r['delta_be']:+10.1f} {r['delta_r']:+8.3f} {r['w_eff']:8.4f}")

    print("\nTarget: |ΔBE| < 50 mHa, ΔR_eq ≈ 0")
