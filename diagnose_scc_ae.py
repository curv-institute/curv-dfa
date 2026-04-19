#!/usr/bin/env uv run
# /// script
# dependencies = ["numpy"]
# ///
"""
Diagnose why SCC v2 hurts atomization energies on G2-1.

Hypothesis: SCC shifts atom energies more than it shifts molecule energies,
so AE = sum(atom) - molecule has the wrong sign of error.

Reads g2_loocv_energies.csv and decomposes the AE error from MLR (α=0.001,
λ=0.05) vs Fisher (α=0.001, λ=0) into atom-side and molecule-side contributions.
"""
from __future__ import annotations
import csv
from pathlib import Path

CACHE = Path(__file__).parent / "g2_loocv_energies.csv"

ATOMS_NEEDED = {"H", "Li", "C", "N", "O", "F", "S", "Cl"}

MOLECULES = {
    "H2":   {"H": 2},
    "LiH":  {"Li": 1, "H": 1},
    "CH4":  {"C": 1, "H": 4},
    "NH3":  {"N": 1, "H": 3},
    "H2O":  {"O": 1, "H": 2},
    "HF":   {"H": 1, "F": 1},
    "N2":   {"N": 2},
    "F2":   {"F": 2},
    "CO":   {"C": 1, "O": 1},
    "HCl":  {"H": 1, "Cl": 1},
    "H2S":  {"S": 1, "H": 2},
    "LiF":  {"Li": 1, "F": 1},
}

# (alpha, lam) keys we need:
FISHER_KEY = (0.001, 0.0)
MLR_KEY = (0.001, 0.05)


def main():
    # Load cache
    refs = {}            # entity -> ref energy
    grid = {}            # entity -> {(alpha, lam): energy}
    with CACHE.open() as f:
        for row in csv.DictReader(f):
            ent = row["entity"]
            e = float(row["energy_ha"]) if row["energy_ha"] else None
            if not row["alpha"]:
                refs[ent] = e
            else:
                a, l = float(row["alpha"]), float(row["lam"])
                grid.setdefault(ent, {})[(a, l)] = e

    print("Per-entity SCC shift (E[PBE+SCC λ=0.05] - E[PBE α=0.001, λ=0]) in mHa:")
    print(f"{'Entity':>8} | {'PBE-like':>14} | {'+SCC λ=0.05':>14} | {'ΔE_SCC':>10}")
    print("-" * 60)
    shifts = {}
    for ent in sorted(set(list(ATOMS_NEEDED) + list(MOLECULES.keys()))):
        e_fisher = grid[ent][FISHER_KEY]
        e_mlr = grid[ent][MLR_KEY]
        delta = (e_mlr - e_fisher) * 1000  # mHa
        shifts[ent] = delta
        kind = "atom" if ent in ATOMS_NEEDED else "mol "
        print(f"{ent:>8} | {e_fisher:14.6f} | {e_mlr:14.6f} | {delta:+10.3f}  ({kind})")

    print(f"\n{'='*72}")
    print("AE shift decomposition: how does adding SCC change atomization energy?")
    print(f"{'='*72}")
    print(f"{'Mol':>6} | {'sum(ΔE_atom)':>14} | {'ΔE_mol':>10} | {'ΔAE_SCC':>10} | "
          f"{'AE error contrib':>18}")
    print("-" * 80)
    for name, count in MOLECULES.items():
        sum_atom_shift = sum(n * shifts[sym] for sym, n in count.items())
        mol_shift = shifts[name]
        # ΔAE_SCC = Δ(sum_atoms - molecule) = sum_atom_shift - mol_shift
        delta_ae = sum_atom_shift - mol_shift
        # If reference AE is positive (more bound = larger), then a positive ΔAE means
        # SCC made AE LARGER (MORE binding predicted than no-SCC version).
        print(f"{name:>6} | {sum_atom_shift:+14.3f} | {mol_shift:+10.3f} | "
              f"{delta_ae:+10.3f} | {delta_ae:+18.3f} mHa")

    # Reference AEs in mHa for comparison
    print(f"\n{'='*72}")
    print("Compared against reference AE error budget:")
    print(f"{'='*72}")
    print(f"{'Mol':>6} | {'AE_ref':>10} | {'AE_PBE-like':>12} | {'AE_MLR':>10} | "
          f"{'PBE err':>9} | {'MLR err':>9}")
    print("-" * 75)
    for name, count in MOLECULES.items():
        ae_ref = (sum(n * refs[s] for s, n in count.items()) - refs[name]) * 1000
        ae_fisher = (sum(n * grid[s][FISHER_KEY] for s, n in count.items())
                     - grid[name][FISHER_KEY]) * 1000
        ae_mlr = (sum(n * grid[s][MLR_KEY] for s, n in count.items())
                  - grid[name][MLR_KEY]) * 1000
        print(f"{name:>6} | {ae_ref:10.3f} | {ae_fisher:12.3f} | {ae_mlr:10.3f} | "
              f"{ae_fisher - ae_ref:+9.3f} | {ae_mlr - ae_ref:+9.3f}")


if __name__ == "__main__":
    main()
