#!/usr/bin/env uv run
# /// script
# dependencies = ["numpy", "scipy"]
# ///
"""
MLR vs Fisher Information: Ablation analysis on H2/H2+ DFT corrections.

Maps the existing curv-dfa composability data onto the MLR-vs-Fisher
head-to-head comparison:

  E_pbe    -> baseline (no information-theoretic correction)
  E_fisher -> Fisher-only (Frieden's framework analog: K+H, no energy term)
  E_scc    -> energy-only (SCC self-coupling cancellation)
  E_both   -> full MLR analog (K+H+E equivalent: Fisher + SCC composed)
  E_hf     -> Hartree-Fock reference (gold standard, taken as ground truth)

For each correction model we measure |E_model - E_ref| across the
dissociation curve. If E_both consistently beats E_fisher, the energy
term is doing work that Fisher information alone cannot replicate.

Reports:
  - Mean absolute error (MAE) per model vs. HF reference
  - Per-radius residuals
  - Paired Wilcoxon signed-rank test (Fisher vs. Both)
  - AIC/BIC corrected for parameter count
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

DATA_DIR = Path(__file__).parent
H2_CSV = DATA_DIR / "composability_h2.csv"
H2PLUS_CSV = DATA_DIR / "composability_h2plus.csv"


def load(csv_path: Path, ref_col: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(csv_path.open()))
    cols = ["R", "E_pbe", "E_fisher", "E_scc", "E_both", ref_col]
    out = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    out["E_ref"] = out.pop(ref_col)
    return out


def residuals(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "PBE":          np.abs(d["E_pbe"]    - d["E_ref"]),
        "Fisher":       np.abs(d["E_fisher"] - d["E_ref"]),
        "SCC (E only)": np.abs(d["E_scc"]    - d["E_ref"]),
        "MLR (K+H+E)":  np.abs(d["E_both"]   - d["E_ref"]),
    }


def aic_bic(residuals_ha: np.ndarray, n_params: int) -> tuple[float, float]:
    """Gaussian-likelihood AIC/BIC from squared residuals. Lower is better."""
    n = len(residuals_ha)
    rss = float(np.sum(residuals_ha ** 2))
    sigma2 = rss / n
    log_lik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1)
    aic = 2 * n_params - 2 * log_lik
    bic = n_params * math.log(n) - 2 * log_lik
    return aic, bic


N_PARAMS = {
    "PBE":          0,  # no calibrated parameter
    "Fisher":       1,  # alpha
    "SCC (E only)": 1,  # lambda
    "MLR (K+H+E)":  2,  # alpha + lambda
}


def report(name: str, d: dict[str, np.ndarray]) -> dict:
    res = residuals(d)
    print(f"\n{'='*72}")
    print(f"SYSTEM: {name}")
    print(f"{'='*72}")
    print(f"Reference: HF/UHF, {len(d['R'])} bond distances")

    print(f"\nPer-radius |E_model - E_HF| (Hartree):")
    print(f"{'R(A)':>6} | " + " | ".join(f"{m:>12}" for m in res))
    print("-" * (8 + 15 * len(res)))
    for i, R in enumerate(d["R"]):
        row = f"{R:6.3f} | " + " | ".join(f"{res[m][i]:12.6f}" for m in res)
        print(row)

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

    # Head-to-head: Fisher vs MLR(K+H+E)
    print(f"\nHead-to-head: Fisher vs MLR (K+H+E)")
    print("-" * 60)
    fisher_r = res["Fisher"]
    mlr_r = res["MLR (K+H+E)"]
    diffs = fisher_r - mlr_r  # positive => MLR beats Fisher
    n_mlr_wins = int(np.sum(diffs > 0))
    print(f"  Fisher MAE:      {np.mean(fisher_r):.6f} Ha")
    print(f"  MLR    MAE:      {np.mean(mlr_r):.6f} Ha")
    print(f"  Improvement:     {(np.mean(fisher_r) - np.mean(mlr_r))*1000:+.3f} mHa")
    print(f"  MLR wins at:     {n_mlr_wins}/{len(diffs)} radii")
    print(f"  ΔAIC (Fisher-MLR): {summary['Fisher']['aic'] - summary['MLR (K+H+E)']['aic']:+.2f}"
          f"   (positive => MLR preferred even after parameter penalty)")
    print(f"  ΔBIC (Fisher-MLR): {summary['Fisher']['bic'] - summary['MLR (K+H+E)']['bic']:+.2f}")

    if len(diffs) >= 6:
        try:
            stat, pval = stats.wilcoxon(fisher_r, mlr_r, alternative="greater")
            print(f"  Wilcoxon (Fisher > MLR): W={stat:.2f}, p={pval:.4f}")
        except ValueError as e:
            print(f"  Wilcoxon: {e}")
    else:
        print(f"  Wilcoxon: skipped (n={len(diffs)} < 6)")

    return summary


def main() -> None:
    h2 = load(H2_CSV, "E_hf")
    h2plus = load(H2PLUS_CSV, "E_uhf")

    s_h2 = report("H2 (RKS, HF reference)", h2)
    s_h2p = report("H2+ (UKS, UHF reference)", h2plus)

    print(f"\n{'='*72}")
    print("OVERALL VERDICT")
    print(f"{'='*72}")
    for name, s in [("H2", s_h2), ("H2+", s_h2p)]:
        fisher_mae = s["Fisher"]["mae"]
        mlr_mae = s["MLR (K+H+E)"]["mae"]
        d_bic = s["Fisher"]["bic"] - s["MLR (K+H+E)"]["bic"]
        if mlr_mae < fisher_mae and d_bic > 0:
            verdict = "MLR > Fisher"
        elif mlr_mae > fisher_mae:
            verdict = "Fisher > MLR"
        else:
            verdict = "INCONCLUSIVE"
        print(f"  {name:>4}: Fisher MAE={fisher_mae:.6f} Ha, "
              f"MLR MAE={mlr_mae:.6f} Ha, ΔBIC={d_bic:+.2f}  [{verdict}]")


if __name__ == "__main__":
    main()
