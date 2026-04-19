#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy", "scipy"]
# ///
"""
MLR vs Fisher Information: Ablation v2 with proper SCC and correlated reference.

Fixes two methodological flaws in v1:
  1. v1 used HF as the "truth" reference. HF lacks correlation energy, so
     "distance to HF" did not measure distance to truth for H2. This run uses
     CCSD(T) for H2 (essentially exact in def2-svp) and UHF for H2+ (exact for
     a one-electron system at dissociation).
  2. v1 used CSV data generated with the inline SCC v1 implementation in
     composability_test.py, which has no density-suppression gate h(ρ).
     SCC v1 over-corrects in many-electron regions. This run uses SCC v2 from
     rift_functionals.py (SCCModule), which gates with h(ρ) = 1/(1+(ρ/ρ_s)^p).

Mapping to MLR:
  PBE           -> baseline (no info-theoretic correction)
  Fisher (K+H)  -> alpha-coupled Fisher info + harmonizer (geometry/density)
  SCC v2 (E)    -> lam-coupled self-Coulomb cancellation with density gate
  MLR (K+H+E)   -> Fisher + SCC v2 composed

Calibration uses experiments.py defaults that were tuned for the broader
CO/NO + H2+ balance, not cherry-picked for this comparison:
  H2+: alpha=0.006, lam=0.2
  H2:  alpha=0.005, lam=0.2
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
from pyscf import cc, dft, gto, scf
from scipy import stats

from rift_functionals import (
    BASE_XC, FisherCfg, FisherModule, SCCCfg, SCCModule,
)

# Calibration constants from experiments.py (not tuned for this test)
ALPHA_H2PLUS = 0.006
LAM_H2PLUS = 0.2
ALPHA_H2 = 0.005
LAM_H2 = 0.2

DISTANCES_H2PLUS = [0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
DISTANCES_H2 = [0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0]

OUT_DIR = Path(__file__).parent
H2_OUT = OUT_DIR / "ablation_v2_h2.csv"
H2PLUS_OUT = OUT_DIR / "ablation_v2_h2plus.csv"


def _make_uks(mol, R):
    mf = dft.UKS(mol)
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 150
    if R > 2.0:
        mf.level_shift = 0.2
        mf.damp = 0.3
    return mf


def _make_rks(mol, R):
    mf = dft.RKS(mol)
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 150
    if R > 2.0:
        mf.level_shift = 0.2
        mf.damp = 0.3
    return mf


def h2plus_energies(R, alpha, lam):
    """Return dict of energies for one H2+ bond distance."""
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {R}", basis="def2-svp",
                charge=1, spin=1, verbose=0)
    out = {}

    # PBE baseline
    mf = _make_uks(mol, R)
    mf.xc = BASE_XC
    mf.kernel()
    out["E_pbe"] = mf.e_tot

    # Fisher only (K+H), no SCC
    mf = _make_uks(mol, R)
    FisherModule(FisherCfg(alpha=alpha)).attach(mf)
    mf.kernel()
    out["E_fisher"] = mf.e_tot

    # SCC v2 only (E), starting from PBE density
    mf = _make_uks(mol, R)
    mf.xc = BASE_XC
    mf.kernel()
    if mf.converged:
        dm0 = mf.make_rdm1()
        SCCModule(SCCCfg(lam=lam)).attach(mf)
        mf.kernel(dm0=dm0)
    out["E_scc"] = mf.e_tot

    # MLR analog: Fisher + SCC v2
    mf = _make_uks(mol, R)
    FisherModule(FisherCfg(alpha=alpha)).attach(mf)
    mf.kernel()
    if mf.converged:
        dm0 = mf.make_rdm1()
        SCCModule(SCCCfg(lam=lam)).attach(mf)
        mf.kernel(dm0=dm0)
    out["E_both"] = mf.e_tot

    # UHF reference (exact for 1 electron)
    mf = scf.UHF(mol)
    mf.diis_space = 12
    mf.max_cycle = 150
    if R > 2.0:
        mf.level_shift = 0.2
        mf.damp = 0.3
    mf.kernel()
    out["E_ref"] = mf.e_tot

    return out


def h2_energies(R, alpha, lam):
    """Return dict of energies for one H2 bond distance."""
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {R}", basis="def2-svp",
                charge=0, spin=0, verbose=0)
    out = {}

    # PBE baseline
    mf = _make_rks(mol, R)
    mf.xc = BASE_XC
    mf.kernel()
    out["E_pbe"] = mf.e_tot

    # Fisher only
    mf = _make_rks(mol, R)
    FisherModule(FisherCfg(alpha=alpha)).attach(mf)
    mf.kernel()
    out["E_fisher"] = mf.e_tot

    # SCC v2 only
    mf = _make_rks(mol, R)
    mf.xc = BASE_XC
    mf.kernel()
    if mf.converged:
        dm0 = mf.make_rdm1()
        SCCModule(SCCCfg(lam=lam)).attach(mf)
        mf.kernel(dm0=dm0)
    out["E_scc"] = mf.e_tot

    # MLR analog: Fisher + SCC v2
    mf = _make_rks(mol, R)
    FisherModule(FisherCfg(alpha=alpha)).attach(mf)
    mf.kernel()
    if mf.converged:
        dm0 = mf.make_rdm1()
        SCCModule(SCCCfg(lam=lam)).attach(mf)
        mf.kernel(dm0=dm0)
    out["E_both"] = mf.e_tot

    # CCSD(T) reference (essentially exact for H2/def2-svp)
    mf_hf = scf.RHF(mol)
    mf_hf.kernel()
    cc_obj = cc.CCSD(mf_hf).run()
    et = cc_obj.ccsd_t()
    out["E_ref"] = mf_hf.e_tot + cc_obj.e_corr + et

    return out


def aic_bic(residuals_ha, n_params):
    n = len(residuals_ha)
    rss = float(np.sum(residuals_ha ** 2))
    sigma2 = rss / n
    log_lik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1)
    aic = 2 * n_params - 2 * log_lik
    bic = n_params * math.log(n) - 2 * log_lik
    return aic, bic


N_PARAMS = {
    "PBE":          0,
    "Fisher":       1,
    "SCC v2":       1,
    "MLR (K+H+E)":  2,
}


def report(name, ref_label, rows):
    R = np.array([r["R"] for r in rows])
    res = {
        "PBE":         np.abs(np.array([r["E_pbe"]    for r in rows]) - np.array([r["E_ref"] for r in rows])),
        "Fisher":      np.abs(np.array([r["E_fisher"] for r in rows]) - np.array([r["E_ref"] for r in rows])),
        "SCC v2":      np.abs(np.array([r["E_scc"]    for r in rows]) - np.array([r["E_ref"] for r in rows])),
        "MLR (K+H+E)": np.abs(np.array([r["E_both"]   for r in rows]) - np.array([r["E_ref"] for r in rows])),
    }

    print(f"\n{'='*72}")
    print(f"SYSTEM: {name}    Reference: {ref_label}")
    print(f"{'='*72}")

    print(f"\nPer-radius |E_model - E_ref| (Hartree):")
    print(f"{'R(A)':>6} | " + " | ".join(f"{m:>12}" for m in res))
    print("-" * (8 + 15 * len(res)))
    for i, r in enumerate(R):
        print(f"{r:6.3f} | " + " | ".join(f"{res[m][i]:12.6f}" for m in res))

    print(f"\nSummary statistics (Hartree):")
    print(f"{'Model':>14}  {'MAE':>10}  {'Max err':>10}  {'AIC':>10}  {'BIC':>10}")
    print("-" * 60)
    summary = {}
    for m, r in res.items():
        mae = float(np.mean(r))
        mx = float(np.max(r))
        aic, bic = aic_bic(r, N_PARAMS[m])
        summary[m] = {"mae": mae, "max": mx, "aic": aic, "bic": bic}
        print(f"{m:>14}  {mae:10.6f}  {mx:10.6f}  {aic:10.2f}  {bic:10.2f}")

    fisher_r = res["Fisher"]
    mlr_r = res["MLR (K+H+E)"]
    diffs = fisher_r - mlr_r
    n_mlr_wins = int(np.sum(diffs > 0))
    print(f"\nHead-to-head: Fisher vs MLR (K+H+E)")
    print("-" * 60)
    print(f"  Fisher MAE:      {np.mean(fisher_r):.6f} Ha")
    print(f"  MLR    MAE:      {np.mean(mlr_r):.6f} Ha")
    print(f"  Improvement:     {(np.mean(fisher_r) - np.mean(mlr_r))*1000:+.3f} mHa  (positive = MLR better)")
    print(f"  MLR wins at:     {n_mlr_wins}/{len(diffs)} radii")
    print(f"  ΔAIC (Fisher-MLR): {summary['Fisher']['aic'] - summary['MLR (K+H+E)']['aic']:+.2f}")
    print(f"  ΔBIC (Fisher-MLR): {summary['Fisher']['bic'] - summary['MLR (K+H+E)']['bic']:+.2f}")
    if len(diffs) >= 6:
        try:
            stat, pval = stats.wilcoxon(fisher_r, mlr_r, alternative="greater")
            print(f"  Wilcoxon (Fisher > MLR): W={stat:.2f}, p={pval:.4f}")
        except ValueError as e:
            print(f"  Wilcoxon: {e}")
    return summary


def main():
    print(f"Calibration: alpha_H2+={ALPHA_H2PLUS}, lam_H2+={LAM_H2PLUS}, "
          f"alpha_H2={ALPHA_H2}, lam_H2={LAM_H2}")
    print("(default values from experiments.py:run_composability_test, "
          "tuned for CO/NO + H2+ balance)\n")

    print("Computing H2+ energies...")
    rows_h2plus = []
    for R in DISTANCES_H2PLUS:
        e = h2plus_energies(R, ALPHA_H2PLUS, LAM_H2PLUS)
        e["R"] = R
        rows_h2plus.append(e)
        print(f"  R={R:.2f} A done")

    print("\nComputing H2 energies...")
    rows_h2 = []
    for R in DISTANCES_H2:
        e = h2_energies(R, ALPHA_H2, LAM_H2)
        e["R"] = R
        rows_h2.append(e)
        print(f"  R={R:.2f} A done")

    # Save CSVs
    for path, rows in [(H2PLUS_OUT, rows_h2plus), (H2_OUT, rows_h2)]:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["R", "E_pbe", "E_fisher", "E_scc", "E_both", "E_ref"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})

    s_h2p = report("H2+ (UKS)", "UHF (exact for 1 electron)", rows_h2plus)
    s_h2 = report("H2 (RKS)", "CCSD(T) (essentially exact in def2-svp)", rows_h2)

    print(f"\n{'='*72}")
    print("OVERALL VERDICT (vs proper correlated reference)")
    print(f"{'='*72}")
    for name, s in [("H2+", s_h2p), ("H2 ", s_h2)]:
        fisher_mae = s["Fisher"]["mae"]
        mlr_mae = s["MLR (K+H+E)"]["mae"]
        d_bic = s["Fisher"]["bic"] - s["MLR (K+H+E)"]["bic"]
        if mlr_mae < fisher_mae and d_bic > 0:
            verdict = "MLR > Fisher"
        elif mlr_mae > fisher_mae:
            verdict = "Fisher > MLR"
        else:
            verdict = "INCONCLUSIVE"
        print(f"  {name}: Fisher MAE={fisher_mae:.6f} Ha, "
              f"MLR MAE={mlr_mae:.6f} Ha, ΔBIC={d_bic:+.2f}  [{verdict}]")
    print(f"\nResults written to {H2PLUS_OUT.name} and {H2_OUT.name}")


if __name__ == "__main__":
    main()
