#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy", "scipy"]
# ///
"""
MLR vs Fisher head-to-head with per-system optimal calibration.

The v2 ablation showed Fisher beats MLR at default calibration (α, λ tuned for
CO/NO+H2+ balance). That isn't a fair test — it punishes MLR for using
parameters not optimized for these systems. This run gives each model its
best shot:

  Fisher: grid-search α to minimize MAE vs reference (single parameter)
  MLR:    grid-search (α, λ) jointly to minimize MAE (two parameters)

If MLR still loses on BIC after both models are optimally calibrated, the
energy term is genuinely not earning its place on this benchmark. If MLR
wins, the v2 result was a calibration artifact.

Caveat: this is in-sample optimization (calibrate and evaluate on the same
points). With only 7-8 radii per system, there is no clean train/test split
to do here. Treat the result as an upper bound on each model's accuracy on
this data.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
from pyscf import cc, dft, gto, scf

from rift_functionals import (
    BASE_XC, FisherCfg, FisherModule, SCCCfg, SCCModule,
)

DISTANCES_H2PLUS = [0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
DISTANCES_H2 = [0.5, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0]

ALPHA_GRID = [0.001, 0.002, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05]
LAM_GRID   = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

OUT_DIR = Path(__file__).parent


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


def h2plus_mf(R):
    return gto.M(atom=f"H 0 0 0; H 0 0 {R}", basis="def2-svp",
                 charge=1, spin=1, verbose=0)


def h2_mf(R):
    return gto.M(atom=f"H 0 0 0; H 0 0 {R}", basis="def2-svp",
                 charge=0, spin=0, verbose=0)


def energy_mlr(mol, R, alpha, lam, uks=True):
    """Run PBE + Fisher (alpha) + SCC v2 (lam). lam=0 disables SCC."""
    mf = _make_uks(mol, R) if uks else _make_rks(mol, R)
    if alpha > 0:
        FisherModule(FisherCfg(alpha=alpha)).attach(mf)
    else:
        mf.xc = BASE_XC
    mf.kernel()
    if lam > 0 and mf.converged:
        dm0 = mf.make_rdm1()
        SCCModule(SCCCfg(lam=lam)).attach(mf)
        mf.kernel(dm0=dm0)
    return mf.e_tot, bool(mf.converged)


def reference_energy(R, system):
    if system == "h2plus":
        mol = h2plus_mf(R)
        mf = scf.UHF(mol)
        mf.diis_space = 12
        mf.max_cycle = 150
        if R > 2.0:
            mf.level_shift = 0.2
            mf.damp = 0.3
        mf.kernel()
        return mf.e_tot
    else:
        mol = h2_mf(R)
        mf_hf = scf.RHF(mol)
        mf_hf.kernel()
        cc_obj = cc.CCSD(mf_hf).run()
        et = cc_obj.ccsd_t()
        return mf_hf.e_tot + cc_obj.e_corr + et


def aic_bic(residuals_ha, n_params):
    n = len(residuals_ha)
    rss = float(np.sum(residuals_ha ** 2))
    sigma2 = rss / n if rss > 0 else 1e-30
    log_lik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1)
    return 2 * n_params - 2 * log_lik, n_params * math.log(n) - 2 * log_lik


def calibrate_fisher(distances, system, refs):
    """Grid search alpha to minimize MAE."""
    uks = (system == "h2plus")
    best = None
    energies_at_best = None
    print(f"  Fisher α scan: ", end="", flush=True)
    for alpha in ALPHA_GRID:
        es = []
        for R in distances:
            mol = h2plus_mf(R) if uks else h2_mf(R)
            e, conv = energy_mlr(mol, R, alpha, 0.0, uks=uks)
            if not conv:
                es = None
                break
            es.append(e)
        if es is None:
            print(f"α={alpha}:fail ", end="", flush=True)
            continue
        mae = float(np.mean(np.abs(np.array(es) - refs)))
        print(f"α={alpha}:{mae*1000:.1f} ", end="", flush=True)
        if best is None or mae < best[1]:
            best = (alpha, mae)
            energies_at_best = es
    print()
    if best is None:
        raise RuntimeError("Fisher calibration failed — every α diverged")
    return best[0], best[1], energies_at_best


def calibrate_mlr(distances, system, refs):
    """Grid search (alpha, lam) jointly to minimize MAE."""
    uks = (system == "h2plus")
    best = None
    energies_at_best = None
    print(f"  MLR (α,λ) joint scan...")
    for alpha in ALPHA_GRID:
        for lam in LAM_GRID:
            if lam == 0.0:
                continue  # that's just Fisher
            es = []
            for R in distances:
                mol = h2plus_mf(R) if uks else h2_mf(R)
                e, conv = energy_mlr(mol, R, alpha, lam, uks=uks)
                if not conv:
                    es = None
                    break
                es.append(e)
            if es is None:
                continue
            mae = float(np.mean(np.abs(np.array(es) - refs)))
            if best is None or mae < best[2]:
                best = (alpha, lam, mae)
                energies_at_best = es
    if best is not None:
        print(f"    best: α={best[0]}, λ={best[1]}, MAE={best[2]*1000:.2f} mHa")
    return best, energies_at_best


def main():
    print("Computing reference energies...")
    refs_h2plus = np.array([reference_energy(R, "h2plus") for R in DISTANCES_H2PLUS])
    refs_h2     = np.array([reference_energy(R, "h2")     for R in DISTANCES_H2])
    print(f"  H2+ UHF references: {len(refs_h2plus)} points")
    print(f"  H2 CCSD(T) references: {len(refs_h2)} points")

    results = {}
    for name, distances, refs, system in [
        ("H2+", DISTANCES_H2PLUS, refs_h2plus, "h2plus"),
        ("H2",  DISTANCES_H2,     refs_h2,     "h2"),
    ]:
        print(f"\n{'='*72}")
        print(f"CALIBRATING ON {name}")
        print(f"{'='*72}")

        a_fisher, mae_fisher, e_fisher = calibrate_fisher(distances, system, refs)
        best_mlr, e_mlr = calibrate_mlr(distances, system, refs)
        if best_mlr is None or e_mlr is None:
            raise RuntimeError(f"MLR calibration failed for {name} — all (α,λ) diverged")
        a_mlr, l_mlr, mae_mlr = best_mlr

        res_fisher = np.abs(np.array(e_fisher) - refs)
        res_mlr = np.abs(np.array(e_mlr) - refs)
        aic_f, bic_f = aic_bic(res_fisher, 1)
        aic_m, bic_m = aic_bic(res_mlr, 2)

        results[name] = {
            "fisher": {"alpha": a_fisher, "mae": mae_fisher, "aic": aic_f, "bic": bic_f},
            "mlr":    {"alpha": a_mlr, "lam": l_mlr, "mae": mae_mlr, "aic": aic_m, "bic": bic_m},
            "n": len(distances),
        }

        print(f"\n  Best Fisher: α={a_fisher}, MAE={mae_fisher*1000:.2f} mHa, AIC={aic_f:.2f}, BIC={bic_f:.2f}")
        print(f"  Best MLR:    α={a_mlr}, λ={l_mlr}, MAE={mae_mlr*1000:.2f} mHa, AIC={aic_m:.2f}, BIC={bic_m:.2f}")
        print(f"  ΔBIC (Fisher-MLR): {bic_f - bic_m:+.2f}  (positive => MLR preferred)")

    # Save calibration table
    out = OUT_DIR / "calibrated_v2_results.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "n_radii", "fisher_alpha", "fisher_mae_ha", "fisher_aic", "fisher_bic",
                    "mlr_alpha", "mlr_lam", "mlr_mae_ha", "mlr_aic", "mlr_bic", "delta_bic_pos_means_mlr"])
        for name, r in results.items():
            w.writerow([name, r["n"],
                        r["fisher"]["alpha"], r["fisher"]["mae"], r["fisher"]["aic"], r["fisher"]["bic"],
                        r["mlr"]["alpha"], r["mlr"]["lam"], r["mlr"]["mae"], r["mlr"]["aic"], r["mlr"]["bic"],
                        r["fisher"]["bic"] - r["mlr"]["bic"]])

    print(f"\n{'='*72}")
    print("OPTIMALLY-CALIBRATED HEAD-TO-HEAD (in-sample)")
    print(f"{'='*72}")
    print(f"{'System':>8} | {'Fisher MAE':>12} | {'MLR MAE':>12} | "
          f"{'ΔBIC':>8} | {'Verdict':>14}")
    print("-" * 70)
    for name, r in results.items():
        d_bic = r["fisher"]["bic"] - r["mlr"]["bic"]
        if r["mlr"]["mae"] < r["fisher"]["mae"] and d_bic > 2:  # BIC>2 = positive evidence
            verdict = "MLR > Fisher"
        elif r["mlr"]["mae"] < r["fisher"]["mae"]:
            verdict = "MLR ≈ Fisher"
        else:
            verdict = "Fisher > MLR"
        print(f"{name:>8} | {r['fisher']['mae']*1000:8.2f} mHa | "
              f"{r['mlr']['mae']*1000:8.2f} mHa | {d_bic:+8.2f} | {verdict:>14}")
    print(f"\nResults written to {out.name}")


if __name__ == "__main__":
    main()
