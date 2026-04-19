#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy", "scipy"]
# ///
"""
K19 scaling-law test: does λ_SCC behave like K19's λ?

K19 claims S ∝ λ⁻¹ with R² = 1.000 as a universal scaling law across MLR
domains. If λ_SCC is a member of that family, the entropy of the SCC-corrected
density should scale as 1/λ_SCC.

This script computes the Shannon entropy S = -∫ ρ ln ρ dV of the SCC-corrected
density on each atom and molecule, at each (α, λ) pair from our cached grid.
Then it fits S vs 1/λ and reports R².

Two complementary tests:
  (a) Per-entity test: for each atom/molecule, fit S(λ) at fixed α and check
      R² of S vs 1/λ.
  (b) Cross-entity test: S(λ) should vary across entities but the *scaling
      with λ* should follow the same 1/λ power if K19 is universal.

We hold α fixed at 0.001 (the smallest non-zero value, where Fisher correction
is minimal) so the variation is dominated by SCC.
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
from pyscf import dft, gto
from scipy import stats

from rift_functionals import BASE_XC, FisherCfg, FisherModule, SCCCfg, SCCModule

OUT_DIR = Path(__file__).parent
OUT_CSV = OUT_DIR / "k19_scaling_results.csv"

ATOMS = {"H": 1, "Li": 1, "C": 2, "N": 3, "O": 2, "F": 1, "S": 2, "Cl": 1}

MOLECULES = {
    "H2":   ("H 0 0 0; H 0 0 0.741",       0, 0),
    "LiH":  ("Li 0 0 0; H 0 0 1.595",      0, 0),
    "CH4":  ("C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629", 0, 0),
    "NH3":  ("N 0 0 0.112; H 0 0.939 -0.260; H 0.813 -0.470 -0.260; H -0.813 -0.470 -0.260", 0, 0),
    "H2O":  ("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", 0, 0),
    "HF":   ("H 0 0 0; F 0 0 0.917",       0, 0),
    "N2":   ("N 0 0 0; N 0 0 1.098",       0, 0),
    "F2":   ("F 0 0 0; F 0 0 1.412",       0, 0),
    "CO":   ("C 0 0 0; O 0 0 1.128",       0, 0),
    "HCl":  ("H 0 0 0; Cl 0 0 1.275",      0, 0),
    "H2S":  ("S 0 0 0; H 0 0.961 0.929; H 0 -0.961 0.929", 0, 0),
    "LiF":  ("Li 0 0 0; F 0 0 1.564",      0, 0),
}

# K19 test: vary lam at fixed (small) alpha
ALPHA = 0.001
LAM_GRID = [0.05, 0.1, 0.2]  # exclude 0 (no SCC, undefined for this test)


def compute_density_entropy(mf):
    """S = -∫ ρ ln ρ dV, with floor for numerical stability."""
    mol = mf.mol
    grids = mf.grids
    if grids.coords is None:
        grids.build()
    ao = mf._numint.eval_ao(mol, grids.coords, deriv=0)
    dm = mf.make_rdm1()
    dm_arr = np.asarray(dm)
    if dm_arr.ndim == 3:  # UKS
        rho = mf._numint.eval_rho(mol, ao, dm_arr[0] + dm_arr[1], xctype="LDA")
    else:
        rho = mf._numint.eval_rho(mol, ao, dm_arr, xctype="LDA")
    rho = np.maximum(rho, 1e-12)
    n_el = float(np.sum(grids.weights * rho))
    p = rho / n_el  # normalize to probability density
    # Differential entropy
    return float(-np.sum(grids.weights * p * np.log(p)) * n_el)


def run_one(atom_str, charge, spin, alpha, lam):
    mol = gto.M(atom=atom_str, basis="def2-svp",
                charge=charge, spin=spin, verbose=0)
    is_open = spin > 0
    mf = (dft.UKS if is_open else dft.RKS)(mol)
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 200
    mf.level_shift = 0.1
    mf.damp = 0.2
    if alpha > 0:
        FisherModule(FisherCfg(alpha=alpha)).attach(mf)
    else:
        mf.xc = BASE_XC
    mf.kernel()
    if not mf.converged:
        return None
    if lam > 0:
        dm0 = mf.make_rdm1()
        SCCModule(SCCCfg(lam=lam)).attach(mf)
        mf.kernel(dm0=dm0)
        if not mf.converged:
            return None
    return compute_density_entropy(mf)


def main():
    print(f"K19 scaling test: S(λ) at α = {ALPHA}, λ ∈ {LAM_GRID}\n")
    rows = []

    print("Computing entropies for atoms...")
    for sym, sp in ATOMS.items():
        for lam in LAM_GRID:
            S = run_one(f"{sym} 0 0 0", 0, sp, ALPHA, lam)
            rows.append({"entity": sym, "kind": "atom", "lam": lam,
                        "inv_lam": 1.0 / lam, "S": S})
            print(f"  {sym}  λ={lam:.3f}  S={S:+.4f}" if S is not None else f"  {sym}  λ={lam:.3f}  FAIL")

    print("\nComputing entropies for molecules...")
    for name, (astr, ch, sp) in MOLECULES.items():
        for lam in LAM_GRID:
            S = run_one(astr, ch, sp, ALPHA, lam)
            rows.append({"entity": name, "kind": "molecule", "lam": lam,
                        "inv_lam": 1.0 / lam, "S": S})
            print(f"  {name}  λ={lam:.3f}  S={S:+.4f}" if S is not None else f"  {name}  λ={lam:.3f}  FAIL")

    # Save raw
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entity", "kind", "lam", "inv_lam", "S"])
        w.writeheader()
        w.writerows(rows)

    # Per-entity fit: S vs 1/λ
    print(f"\n{'='*72}")
    print("Per-entity linear fit: S = m · (1/λ) + b")
    print(f"{'='*72}")
    print(f"{'Entity':>8} | {'kind':>9} | {'slope':>10} | {'intercept':>10} | "
          f"{'R²':>8} | {'K19 fit?':>10}")
    print("-" * 70)
    entities = sorted({r["entity"] for r in rows})
    fit_results = []
    for ent in entities:
        ent_rows = [r for r in rows if r["entity"] == ent and r["S"] is not None]
        if len(ent_rows) < 3:
            continue
        x = np.array([r["inv_lam"] for r in ent_rows])
        y = np.array([r["S"] for r in ent_rows])
        result = stats.linregress(x, y)
        slope = float(result.slope)  # type: ignore[attr-defined]
        intercept = float(result.intercept)  # type: ignore[attr-defined]
        r2 = float(result.rvalue) ** 2  # type: ignore[attr-defined]
        p_value = float(result.pvalue)  # type: ignore[attr-defined]
        kind = ent_rows[0]["kind"]
        verdict = "✓" if r2 > 0.95 else ("~" if r2 > 0.8 else "✗")
        print(f"{ent:>8} | {kind:>9} | {slope:+10.4f} | {intercept:+10.4f} | "
              f"{r2:8.4f} | {verdict:>10}")
        fit_results.append({"entity": ent, "kind": kind, "slope": slope,
                           "intercept": intercept, "r2": r2, "p": p_value})

    # Aggregate
    print(f"\n{'='*72}")
    print("AGGREGATE")
    print(f"{'='*72}")
    r2_atoms = [r["r2"] for r in fit_results if r["kind"] == "atom"]
    r2_mols = [r["r2"] for r in fit_results if r["kind"] == "molecule"]
    print(f"  Atoms     mean R² = {np.mean(r2_atoms):.4f}, "
          f"median = {np.median(r2_atoms):.4f}, n = {len(r2_atoms)}")
    print(f"  Molecules mean R² = {np.mean(r2_mols):.4f}, "
          f"median = {np.median(r2_mols):.4f}, n = {len(r2_mols)}")
    print(f"  All       mean R² = {np.mean([r['r2'] for r in fit_results]):.4f}")
    n_strong = sum(1 for r in fit_results if r["r2"] > 0.95)
    n_total = len(fit_results)
    print(f"  Entities with R² > 0.95: {n_strong}/{n_total}  "
          f"({100*n_strong/n_total:.0f}%)")

    print(f"\n  Reminder: K19 claims R² = 1.000 (perfect linearity).")
    print(f"  If our R² are close to 1, λ_SCC behaves like K19's λ.")
    print(f"  If much lower, λ_SCC is structurally different from the K19 family.")

    print(f"\nResults: {OUT_CSV.name}")


if __name__ == "__main__":
    main()
