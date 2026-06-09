#!/usr/bin/env python3
"""
RIFT Composability Test: Fisher + SCC on H2+ and H2 — SCC v2 regeneration.

This is a minimal port of composability_test.py (which generated the original
composability_h2.csv / composability_h2plus.csv with the retired inline SCC v1).
The protocol — systems, basis (def2-svp), base functional (PBE), grid level,
distance grids, calibration procedure (alpha target +15 mHa, lambda target
-10 mHa at the reference geometry), and the additivity metric
epsilon(R) = dE_FS - (dE_F + dE_S) — is unchanged.

The ONLY substantive change: the Fisher and SCC channels are taken from the
canonical current implementation in rift_functionals.py (FisherModule,
SCCModule). SCCModule is SCC v2: it adds the density-suppression gate
h(rho) = 1/(1+(rho/rho_s)^p) (SCCCfg defaults: rho_s=0.03, h_power=6,
w_power=4), which v1 lacked. v1 was retired because it over-corrects in
many-electron regions.

Outputs: composability_h2plus_sccv2.csv, composability_h2_sccv2.csv
(same schema as the v1 CSVs; v1 files left untouched).
"""

from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyscf import gto, dft, scf

from rift_functionals import (
    BASE_XC,
    FisherCfg,
    FisherModule,
    SCCCfg,
    SCCModule,
)


# =============================================================================
# Single-Point Calculation (same protocol as composability_test.py)
# =============================================================================

@dataclass
class CalcResult:
    R: float
    mode: str
    E: float
    converged: bool


def run_h2plus(R: float, mode: str, alpha: float = 0.0, lam: float = 0.0) -> CalcResult:
    """H2+ single point. Modes: pbe, fisher, scc, both, uhf"""
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
        FisherModule(FisherCfg(alpha=alpha)).attach(mf)
    else:
        mf.xc = BASE_XC

    # First SCF (PBE or Fisher)
    mf.kernel()

    if use_scc and mf.converged:
        dm0 = mf.make_rdm1()
        SCCModule(SCCCfg(lam=lam)).attach(mf)
        mf.kernel(dm0=dm0)

    return CalcResult(R=R, mode=mode, E=mf.e_tot, converged=mf.converged)


def run_h2(R: float, mode: str, alpha: float = 0.0, lam: float = 0.0,
           use_uks: bool = False) -> CalcResult:
    """H2 single point. Modes: pbe, fisher, scc, both, hf"""
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
        FisherModule(FisherCfg(alpha=alpha)).attach(mf)
    else:
        mf.xc = BASE_XC

    mf.kernel()

    if use_scc and mf.converged:
        dm0 = mf.make_rdm1()
        SCCModule(SCCCfg(lam=lam)).attach(mf)
        mf.kernel(dm0=dm0)

    return CalcResult(R=R, mode=mode, E=mf.e_tot, converged=mf.converged)


# =============================================================================
# Calibration (identical procedure to composability_test.py)
# =============================================================================

def calibrate_alpha(system: str, R_ref: float, target_dE_mHa: float = 20.0) -> float:
    """Find alpha that gives target dE at reference geometry."""
    print(f"\n--- Calibrating alpha for {system} at R={R_ref} A ---")
    print(f"Target: dE ~ {target_dE_mHa:+.1f} mHa")

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
        print(f"  alpha={a:.3f}: dE = {de:+.2f} mHa")

    alphas = np.array(list(de_vals.keys()))
    des = np.array(list(de_vals.values()))
    alpha_cal = np.interp(target_dE_mHa, des, alphas)

    print(f"  => Calibrated alpha = {alpha_cal:.4f}")
    return alpha_cal


def calibrate_lambda(system: str, R_ref: float, target_dE_mHa: float = -10.0) -> float:
    """Find lambda that gives target dE at reference geometry."""
    print(f"\n--- Calibrating lambda for {system} at R={R_ref} A ---")
    print(f"Target: dE ~ {target_dE_mHa:+.1f} mHa")

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
        print(f"  lambda={l:.3f}: dE = {de:+.2f} mHa")

    lams = np.array(list(de_vals.keys()))
    des = np.array(list(de_vals.values()))
    lam_cal = np.interp(target_dE_mHa, des, lams)

    print(f"  => Calibrated lambda = {lam_cal:.4f}")
    return lam_cal


# =============================================================================
# Main Test (identical analysis to composability_test.py)
# =============================================================================

def run_composability_test():
    """Run full composability test for Fisher + SCC v2."""

    print("=" * 80)
    print("RIFT COMPOSABILITY TEST: Fisher + SCC v2 (rift_functionals.SCCModule)")
    print("=" * 80)
    print()
    print("SCC v2 gate params (SCCCfg defaults):", SCCCfg().as_dict())
    print()

    # =========================================================================
    # H2+ Test
    # =========================================================================
    print("\n" + "=" * 80)
    print("SYSTEM: H2+ (UKS) - Primary SIE test")
    print("=" * 80)

    R_ref_h2plus = 2.0

    alpha_h2plus = calibrate_alpha("h2plus", R_ref_h2plus, target_dE_mHa=15.0)
    lam_h2plus = calibrate_lambda("h2plus", R_ref_h2plus, target_dE_mHa=-10.0)

    print(f"\nUsing: alpha = {alpha_h2plus:.4f}, lambda = {lam_h2plus:.4f}")

    distances_h2plus = np.array([0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    modes = ["pbe", "fisher", "scc", "both"]

    results_h2plus = {m: [] for m in modes}
    results_h2plus["uhf"] = []

    print(f"\n--- H2+ Dissociation Scan ---")
    print(f"{'R(A)':>6}  {'PBE':>12}  {'Fisher':>12}  {'SCC':>12}  {'Both':>12}  {'UHF':>12}  conv")
    print("-" * 85)

    for R in distances_h2plus:
        row = f"{R:6.3f}"
        conv = ""

        for mode in modes:
            res = run_h2plus(R, mode, alpha=alpha_h2plus, lam=lam_h2plus)
            results_h2plus[mode].append(res)
            row += f"  {res.E:12.8f}"
            conv += "Y" if res.converged else "N"

        res_uhf = run_h2plus(R, "uhf")
        results_h2plus["uhf"].append(res_uhf)
        row += f"  {res_uhf.E:12.8f}"
        conv += "Y" if res_uhf.converged else "N"

        print(row + f"  {conv}")

    print(f"\n--- H2+ Composability Analysis ---")
    print(f"{'R(A)':>6}  {'dE_F':>10}  {'dE_S':>10}  {'dE_FS':>10}  {'eps(R)':>10}  {'eps/(|F|+|S|)':>14}")
    print("-" * 75)

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

        print(f"{R:6.3f}  {dE_F:+10.2f}  {dE_S:+10.2f}  {dE_FS:+10.2f}  {epsilon:+10.2f}  {eps_rel:+14.2%}")

        h2plus_data.append({
            'R': R, 'E_pbe': e_pbe, 'E_fisher': e_f, 'E_scc': e_s, 'E_both': e_fs, 'E_uhf': e_uhf,
            'dE_F': dE_F, 'dE_S': dE_S, 'dE_FS': dE_FS, 'epsilon': epsilon,
            'conv_pbe': results_h2plus["pbe"][i].converged,
            'conv_fisher': results_h2plus["fisher"][i].converged,
            'conv_scc': results_h2plus["scc"][i].converged,
            'conv_both': results_h2plus["both"][i].converged,
        })

    # =========================================================================
    # H2 Test (RKS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SYSTEM: H2 (RKS) - Geometry/bonding test")
    print("=" * 80)

    R_ref_h2 = 0.74

    alpha_h2 = calibrate_alpha("h2", R_ref_h2, target_dE_mHa=15.0)
    lam_h2 = calibrate_lambda("h2", R_ref_h2, target_dE_mHa=-10.0)

    print(f"\nUsing: alpha = {alpha_h2:.4f}, lambda = {lam_h2:.4f}")

    distances_h2 = np.array([0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0])

    results_h2 = {m: [] for m in modes}
    results_h2["hf"] = []

    print(f"\n--- H2 Dissociation Scan ---")
    print(f"{'R(A)':>6}  {'PBE':>12}  {'Fisher':>12}  {'SCC':>12}  {'Both':>12}  {'HF':>12}  conv")
    print("-" * 85)

    for R in distances_h2:
        row = f"{R:6.3f}"
        conv = ""

        for mode in modes:
            res = run_h2(R, mode, alpha=alpha_h2, lam=lam_h2, use_uks=False)
            results_h2[mode].append(res)
            row += f"  {res.E:12.8f}"
            conv += "Y" if res.converged else "N"

        res_hf = run_h2(R, "hf", use_uks=False)
        results_h2["hf"].append(res_hf)
        row += f"  {res_hf.E:12.8f}"
        conv += "Y" if res_hf.converged else "N"

        print(row + f"  {conv}")

    print(f"\n--- H2 Composability Analysis ---")
    print(f"{'R(A)':>6}  {'dE_F':>10}  {'dE_S':>10}  {'dE_FS':>10}  {'eps(R)':>10}  {'eps/(|F|+|S|)':>14}")
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

        print(f"{R:6.3f}  {dE_F:+10.2f}  {dE_S:+10.2f}  {dE_FS:+10.2f}  {epsilon:+10.2f}  {eps_rel:+14.2%}")

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
    print("COMPOSABILITY SUMMARY (SCC v2)")
    print("=" * 80)

    eps_h2plus = [d['epsilon'] for d in h2plus_data]
    dE_sum_h2plus = [abs(d['dE_F']) + abs(d['dE_S']) for d in h2plus_data]
    eps_rel_h2plus = [e/s if s > 0.1 else 0 for e, s in zip(eps_h2plus, dE_sum_h2plus)]

    print(f"\nH2+:")
    print(f"  eps(R) range: {min(eps_h2plus):+.3f} to {max(eps_h2plus):+.3f} mHa")
    print(f"  |eps|/(|dF|+|dS|) max: {max(abs(e) for e in eps_rel_h2plus):.2%}")

    scc_signs_ok = all(d['dE_S'] < 0 for d in h2plus_data if d['R'] >= 1.5)
    both_preserves_scc = all(d['dE_FS'] < d['dE_F'] for d in h2plus_data if d['R'] >= 1.5)

    print(f"  SCC negative at mid-R: {'YES' if scc_signs_ok else 'NO'}")
    print(f"  Combined preserves SCC: {'YES' if both_preserves_scc else 'NO'}")

    eps_h2 = [d['epsilon'] for d in h2_data]
    dE_sum_h2 = [abs(d['dE_F']) + abs(d['dE_S']) for d in h2_data]
    eps_rel_h2 = [e/s if s > 0.1 else 0 for e, s in zip(eps_h2, dE_sum_h2)]

    print(f"\nH2:")
    print(f"  eps(R) range: {min(eps_h2):+.3f} to {max(eps_h2):+.3f} mHa")
    print(f"  |eps|/(|dF|+|dS|) max: {max(abs(e) for e in eps_rel_h2):.2%}")

    print(f"\n" + "-" * 40)
    max_eps_rel = max(max(abs(e) for e in eps_rel_h2plus), max(abs(e) for e in eps_rel_h2))
    print(f"Max relative composability error (both systems): {max_eps_rel:.2%}")

    if max_eps_rel < 0.2 and scc_signs_ok and both_preserves_scc:
        print("VERDICT: COMPOSABLE")
    elif max_eps_rel < 0.5 and scc_signs_ok:
        print("VERDICT: WEAKLY COMPOSABLE")
    else:
        print("VERDICT: NOT COMPOSABLE")

    with open('composability_h2plus_sccv2.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=h2plus_data[0].keys())
        writer.writeheader()
        writer.writerows(h2plus_data)

    with open('composability_h2_sccv2.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=h2_data[0].keys())
        writer.writeheader()
        writer.writerows(h2_data)

    print(f"\nResults written to composability_h2plus_sccv2.csv and composability_h2_sccv2.csv")

    return h2plus_data, h2_data


if __name__ == "__main__":
    run_composability_test()
