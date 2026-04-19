#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy", "scipy"]
# ///
"""
MLR vs Fisher head-to-head on a G2-1 subset with leave-one-out cross-validation.

This is the out-of-sample test the H2/H2+ ablation could not do (only 7-8
points). Here we have 12 small molecules with experimental geometries; for
each held-out molecule we calibrate (α) for Fisher and (α, λ) for MLR on the
remaining 11, then measure atomization-energy error on the held-out one.

Reference: CCSD(T)/def2-svp atomization energies, computed in-script. Not
matched to experimental binding energies — we are asking "how close to the
exact answer in this basis", not "how close to nature".

Output:
  - g2_loocv_results.csv: per-held-out-molecule errors and selected params
  - Console summary: aggregate MAE, ΔBIC equivalent, win rate
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
from pyscf import cc, dft, gto, scf

from rift_functionals import BASE_XC, FisherCfg, FisherModule, SCCCfg, SCCModule

OUT_DIR = Path(__file__).parent
OUT_CSV = OUT_DIR / "g2_loocv_results.csv"
ENERGY_CACHE = OUT_DIR / "g2_loocv_energies.csv"

# Modest grid — 4×4 = 16 (α,λ) combinations
ALPHA_GRID = [0.001, 0.005, 0.01, 0.02]
LAM_GRID = [0.0, 0.05, 0.1, 0.2]

BASIS = "def2-svp"

# Experimental geometries (Å), atoms with spin multiplicity (2S)
ATOMS = {
    # symbol -> spin (2S)
    "H":  1,
    "Li": 1,
    "C":  2,   # ^3P
    "N":  3,   # ^4S
    "O":  2,   # ^3P
    "F":  1,
    "S":  2,   # ^3P
    "Cl": 1,
}

MOLECULES = {
    # name -> (atoms_string, charge, spin, atom_count_dict)
    "H2":   ("H 0 0 0; H 0 0 0.741",       0, 0, {"H": 2}),
    "LiH":  ("Li 0 0 0; H 0 0 1.595",      0, 0, {"Li": 1, "H": 1}),
    "CH4":  ("C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629", 0, 0, {"C": 1, "H": 4}),
    "NH3":  ("N 0 0 0.112; H 0 0.939 -0.260; H 0.813 -0.470 -0.260; H -0.813 -0.470 -0.260", 0, 0, {"N": 1, "H": 3}),
    "H2O":  ("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", 0, 0, {"O": 1, "H": 2}),
    "HF":   ("H 0 0 0; F 0 0 0.917",       0, 0, {"H": 1, "F": 1}),
    "N2":   ("N 0 0 0; N 0 0 1.098",       0, 0, {"N": 2}),
    "F2":   ("F 0 0 0; F 0 0 1.412",       0, 0, {"F": 2}),
    "CO":   ("C 0 0 0; O 0 0 1.128",       0, 0, {"C": 1, "O": 1}),
    "HCl":  ("H 0 0 0; Cl 0 0 1.275",      0, 0, {"H": 1, "Cl": 1}),
    "H2S":  ("S 0 0 0; H 0 0.961 0.929; H 0 -0.961 0.929", 0, 0, {"S": 1, "H": 2}),
    "LiF":  ("Li 0 0 0; F 0 0 1.564",      0, 0, {"Li": 1, "F": 1}),
}


def _make_uks(mol):
    mf = dft.UKS(mol)
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 200
    mf.level_shift = 0.1
    mf.damp = 0.2
    return mf


def _make_rks(mol):
    mf = dft.RKS(mol)
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 200
    mf.level_shift = 0.1
    mf.damp = 0.2
    return mf


def make_mol(atoms_str, charge, spin):
    return gto.M(atom=atoms_str, basis=BASIS, charge=charge, spin=spin, verbose=0)


def make_atom_mol(symbol):
    return gto.M(atom=f"{symbol} 0 0 0", basis=BASIS,
                 charge=0, spin=ATOMS[symbol], verbose=0)


def energy_dft(mol, alpha, lam, force_uks=False):
    """E for PBE+Fisher(alpha)+SCC(lam). lam=0 disables SCC, alpha=0 disables Fisher."""
    is_open_shell = mol.spin > 0 or force_uks
    mf = _make_uks(mol) if is_open_shell else _make_rks(mol)

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

    return float(mf.e_tot)


def reference_energy(mol):
    """CCSD(T) for closed-shell, UCCSD(T) for open-shell."""
    is_open_shell = mol.spin > 0
    if is_open_shell:
        mf = scf.UHF(mol)
    else:
        mf = scf.RHF(mol)
    mf.diis_space = 12
    mf.max_cycle = 200
    mf.level_shift = 0.1
    mf.kernel()
    if not mf.converged:
        return None
    try:
        cc_obj = cc.CCSD(mf).run()
        if not cc_obj.converged:
            return None
        et = cc_obj.ccsd_t()
        return float(mf.e_tot + cc_obj.e_corr + et)
    except Exception as e:
        print(f"    CCSD(T) failed: {e}")
        return None


def build_energy_table():
    """Compute E[molecule_or_atom, alpha_idx, lam_idx] and CCSD(T) refs."""
    print("Computing atom energies (always use UKS for open shells)...")
    atom_energies = {}  # symbol -> {(a_i, l_i): E, "ref": E_ccsdt}
    for sym in ATOMS:
        atom_energies[sym] = {}
        amol = make_atom_mol(sym)
        for ai, a in enumerate(ALPHA_GRID):
            for li, l in enumerate(LAM_GRID):
                e = energy_dft(amol, a, l, force_uks=True)
                atom_energies[sym][(ai, li)] = e
        ref = reference_energy(amol)
        atom_energies[sym]["ref"] = ref
        print(f"  {sym}: ref={ref:.6f}, grid done")

    print("\nComputing molecule energies...")
    mol_energies = {}
    for name, (astr, ch, sp, _) in MOLECULES.items():
        mol_energies[name] = {}
        m = make_mol(astr, ch, sp)
        for ai, a in enumerate(ALPHA_GRID):
            for li, l in enumerate(LAM_GRID):
                e = energy_dft(m, a, l)
                mol_energies[name][(ai, li)] = e
        ref = reference_energy(m)
        mol_energies[name]["ref"] = ref
        print(f"  {name}: ref={ref:.6f}, grid done")

    # Cache as flat CSV
    with ENERGY_CACHE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity", "kind", "alpha", "lam", "energy_ha"])
        for sym, d in atom_energies.items():
            w.writerow([sym, "atom", "", "", d["ref"]])
            for k, e in d.items():
                if k == "ref" or not isinstance(e, float):
                    continue
                ai, li = k
                w.writerow([sym, "atom", ALPHA_GRID[ai], LAM_GRID[li], e])
        for name, d in mol_energies.items():
            w.writerow([name, "molecule", "", "", d["ref"]])
            for k, e in d.items():
                if k == "ref" or not isinstance(e, float):
                    continue
                ai, li = k
                w.writerow([name, "molecule", ALPHA_GRID[ai], LAM_GRID[li], e])

    return atom_energies, mol_energies


def atomization_energy(mol_e, atom_es, atom_count):
    """AE = sum(n_i * E_atom_i) - E_molecule  (positive number, more bound = larger)"""
    if mol_e is None:
        return None
    total_atom = 0.0
    for sym, n in atom_count.items():
        if atom_es[sym] is None:
            return None
        total_atom += n * atom_es[sym]
    return total_atom - mol_e


def all_aes_for_params(atom_energies, mol_energies, ai, li):
    """Compute AE for each molecule at given (alpha, lam) grid index."""
    aes = {}
    for name, (_, _, _, count) in MOLECULES.items():
        atom_es = {sym: atom_energies[sym].get((ai, li)) for sym in count}
        aes[name] = atomization_energy(mol_energies[name].get((ai, li)), atom_es, count)
    return aes


def reference_aes(atom_energies, mol_energies):
    aes = {}
    for name, (_, _, _, count) in MOLECULES.items():
        atom_es = {sym: atom_energies[sym]["ref"] for sym in count}
        aes[name] = atomization_energy(mol_energies[name]["ref"], atom_es, count)
    return aes


def calibrate_on_subset(atom_energies, mol_energies, ref_aes,
                       train_names, mode):
    """
    Find (alpha_i, lam_i) minimizing MAE over train_names.
    mode: "fisher" (lam=0) or "mlr" (lam>0)
    Returns (best_ai, best_li, best_train_mae) or (None, None, None) if all diverge.
    """
    best = None
    for ai in range(len(ALPHA_GRID)):
        for li in range(len(LAM_GRID)):
            if mode == "fisher" and LAM_GRID[li] != 0.0:
                continue
            if mode == "mlr" and LAM_GRID[li] == 0.0:
                continue
            if mode == "fisher" and ALPHA_GRID[ai] == 0.0:
                continue  # would just be PBE
            aes = all_aes_for_params(atom_energies, mol_energies, ai, li)
            errs = []
            for n in train_names:
                if aes[n] is None or ref_aes[n] is None:
                    errs = None
                    break
                errs.append(abs(aes[n] - ref_aes[n]))
            if errs is None:
                continue
            mae = float(np.mean(errs))
            if best is None or mae < best[2]:
                best = (ai, li, mae)
    return best if best else (None, None, None)


def aic_bic_pair(test_errs, n_params, n):
    """Approximate AIC/BIC from residuals. n is total sample count used."""
    rss = float(np.sum(np.array(test_errs) ** 2))
    sigma2 = rss / n if rss > 0 else 1e-30
    log_lik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1)
    return 2 * n_params - 2 * log_lik, n_params * math.log(n) - 2 * log_lik


def main():
    if ENERGY_CACHE.exists():
        print(f"Loading cached energies from {ENERGY_CACHE.name}...")
        atom_energies = {sym: {} for sym in ATOMS}
        mol_energies = {name: {} for name in MOLECULES}
        with ENERGY_CACHE.open() as f:
            r = csv.DictReader(f)
            for row in r:
                ent, kind = row["entity"], row["kind"]
                e = float(row["energy_ha"]) if row["energy_ha"] else None
                target = atom_energies if kind == "atom" else mol_energies
                if not row["alpha"]:
                    target[ent]["ref"] = e
                else:
                    a, l = float(row["alpha"]), float(row["lam"])
                    ai = ALPHA_GRID.index(a)
                    li = LAM_GRID.index(l)
                    target[ent][(ai, li)] = e
        print(f"  loaded {len(atom_energies)} atoms, {len(mol_energies)} molecules")
    else:
        atom_energies, mol_energies = build_energy_table()

    ref_aes = reference_aes(atom_energies, mol_energies)
    print("\nReference atomization energies (Ha):")
    for n, ae in ref_aes.items():
        print(f"  {n:6}: {ae:+.6f}" if ae is not None else f"  {n:6}: FAILED")

    valid_names = [n for n in MOLECULES if ref_aes[n] is not None]
    print(f"\n{len(valid_names)} molecules with valid reference AE\n")

    # PBE baseline (alpha=0, lam=0 → in our grid this is alpha=0.001 with no SCC, but
    # ALPHA_GRID has no 0; so we approximate PBE by smallest alpha with lam=0)
    # Actually for cleanness, compute PBE separately by setting alpha=0 explicitly.
    # We didn't cache that. Use ALPHA_GRID[0] as smallest-perturbation proxy.

    # LOOCV
    print("=" * 72)
    print("LOOCV")
    print("=" * 72)
    print(f"{'Held':>6} | {'F α':>6} | {'F err':>10} | {'M α,λ':>10} | {'M err':>10} | "
          f"{'F-M':>10} | winner")
    print("-" * 80)

    rows_out = []
    fisher_test_errs = []
    mlr_test_errs = []
    fisher_params = []
    mlr_params = []

    for held in valid_names:
        train = [n for n in valid_names if n != held]
        f_ai, f_li, _ = calibrate_on_subset(atom_energies, mol_energies, ref_aes,
                                             train, "fisher")
        m_ai, m_li, _ = calibrate_on_subset(atom_energies, mol_energies, ref_aes,
                                             train, "mlr")
        if f_ai is None or m_ai is None or f_li is None or m_li is None:
            print(f"{held:>6} | calibration failed")
            continue
        assert f_ai is not None and m_ai is not None and f_li is not None and m_li is not None

        # Eval on held-out
        f_aes = all_aes_for_params(atom_energies, mol_energies, f_ai, f_li)
        m_aes = all_aes_for_params(atom_energies, mol_energies, m_ai, m_li)
        held_ref = ref_aes[held]
        if f_aes[held] is None or m_aes[held] is None or held_ref is None:
            print(f"{held:>6} | held-out energy diverged")
            continue

        f_err = abs(f_aes[held] - held_ref)
        m_err = abs(m_aes[held] - held_ref)
        fisher_test_errs.append(f_err)
        mlr_test_errs.append(m_err)
        fisher_params.append(ALPHA_GRID[f_ai])
        mlr_params.append((ALPHA_GRID[m_ai], LAM_GRID[m_li]))
        winner = "MLR" if m_err < f_err else "Fisher"

        print(f"{held:>6} | {ALPHA_GRID[int(f_ai)]:6.3f} | {f_err*1000:8.2f}m | "
              f"{ALPHA_GRID[int(m_ai)]:.3f},{LAM_GRID[int(m_li)]:.2f} | {m_err*1000:8.2f}m | "
              f"{(f_err - m_err)*1000:+8.2f}m | {winner}")

        rows_out.append({
            "held": held,
            "fisher_alpha": ALPHA_GRID[int(f_ai)],
            "fisher_err_ha": f_err,
            "mlr_alpha": ALPHA_GRID[int(m_ai)],
            "mlr_lam": LAM_GRID[int(m_li)],
            "mlr_err_ha": m_err,
            "winner": winner,
        })

    # Save
    if rows_out:
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    # Aggregate
    if fisher_test_errs:
        n = len(fisher_test_errs)
        f_mae = float(np.mean(fisher_test_errs))
        m_mae = float(np.mean(mlr_test_errs))
        f_aic, f_bic = aic_bic_pair(fisher_test_errs, 1, n)
        m_aic, m_bic = aic_bic_pair(mlr_test_errs, 2, n)
        n_mlr_wins = sum(1 for r in rows_out if r["winner"] == "MLR")

        print(f"\n{'='*72}")
        print("LOOCV SUMMARY (out-of-sample)")
        print(f"{'='*72}")
        print(f"  N held-out:   {n}")
        print(f"  Fisher MAE:   {f_mae*1000:.2f} mHa  (= {f_mae*627.5:.2f} kcal/mol)")
        print(f"  MLR    MAE:   {m_mae*1000:.2f} mHa  (= {m_mae*627.5:.2f} kcal/mol)")
        print(f"  Improvement:  {(f_mae - m_mae)*1000:+.2f} mHa  (positive = MLR better)")
        print(f"  MLR wins at:  {n_mlr_wins}/{n} held-out molecules")
        print(f"  ΔAIC (F-M):   {f_aic - m_aic:+.2f}")
        print(f"  ΔBIC (F-M):   {f_bic - m_bic:+.2f}  (positive => MLR preferred even with extra parameter)")
        print(f"\n  Results: {OUT_CSV.name}")
        print(f"  Cache:   {ENERGY_CACHE.name}")


if __name__ == "__main__":
    main()
