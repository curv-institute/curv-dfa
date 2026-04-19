#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy", "scipy"]
# ///
"""
Cross-molecule SIRS test: does T_fit ≈ T_bond hold across G2-1 subset?

H2 result (sirs_h2_first_test.py):
  - H_vN scales exactly as predicted (0 → ln 2)
  - Sign of correction is wrong with naive +λE; should be -λE per F = E - TS
  - With sign flipped, T_fit = 54793 K matches H2 bond energy (52500 K) to 4%

Critical question: is T_fit a chemistry-determined quantity (≈ T_bond), or a
per-system free parameter? If T_fit ≈ T_bond across multiple molecules, the
SIRS reformulation has a real prediction. If T_fit varies arbitrarily, the
SIRS framework is no better than the previous ad-hoc operationalizations.

Test:
  - For each molecule M in G2-1 subset + atoms, compute CCSD(T) 1-RDM
  - Compute H_vN(M) and H_vN(atoms)
  - Compute reference atomization energy AE_ref from CCSD(T)
  - Compute SIRS-corrected AE_sirs = AE_PBE - (H_vN(M) - sum H_vN(atoms)) × k_B T ln 2
  - Find T_fit per molecule that makes AE_sirs match AE_ref
  - Compare T_fit to T_bond = AE_ref / k_B

The signature we're looking for:
  - T_fit ≈ T_bond (constant ratio, R² > 0.95) → SIRS chemistry is real
  - T_fit varies arbitrarily → SIRS is a fit parameter, not a derivation
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
from pyscf import cc, dft, gto, scf

OUT_DIR = Path(__file__).parent
OUT_CSV = OUT_DIR / "sirs_g2_cross_molecule_results.csv"

K_B_HA = 3.166811563e-6  # Ha/K
LN2 = math.log(2)

ATOMS = {"H": 1, "Li": 1, "C": 2, "N": 3, "O": 2, "F": 1, "S": 2, "Cl": 1}

MOLECULES = {
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


def make_mol(atom_str, charge, spin):
    return gto.M(atom=atom_str, basis="def2-svp", charge=charge, spin=spin, verbose=0)


def make_atom_mol(symbol):
    return gto.M(atom=f"{symbol} 0 0 0", basis="def2-svp",
                 charge=0, spin=ATOMS[symbol], verbose=0)


def natural_occupations(dm_ao, mol):
    """Eigenvalues of 1-RDM in Lowdin-orthogonalized basis."""
    s = mol.intor("int1e_ovlp")
    s_eig, s_vec = np.linalg.eigh(s)
    s_half = s_vec @ np.diag(np.sqrt(np.maximum(s_eig, 0))) @ s_vec.T
    if dm_ao.ndim == 3:  # UKS/UHF: spin-resolved
        gamma_orth = s_half @ (dm_ao[0] + dm_ao[1]) @ s_half
    else:
        gamma_orth = s_half @ dm_ao @ s_half
    occs = np.linalg.eigvalsh(gamma_orth)
    return np.sort(occs)[::-1]


def von_neumann_entropy(occs, n_electrons):
    """S_vN = -Σ p_i ln p_i with p_i = occ_i / N."""
    p = occs / float(n_electrons)
    p = p[p > 1e-12]
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def pbe_energy(mol, uks_force=False):
    is_open = mol.spin > 0 or uks_force
    mf = (dft.UKS if is_open else dft.RKS)(mol)
    mf.xc = "GGA_X_PBE,GGA_C_PBE"
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 200
    mf.level_shift = 0.1
    mf.damp = 0.2
    mf.kernel()
    if not mf.converged:
        return None
    return float(mf.e_tot)


def ccsd_t_with_rdm(mol):
    """Run CCSD(T) and return (E_total, H_vN of 1-RDM, n_electrons)."""
    is_open = mol.spin > 0
    if is_open:
        mf = scf.UHF(mol)
    else:
        mf = scf.RHF(mol)
    mf.diis_space = 12
    mf.max_cycle = 200
    mf.level_shift = 0.1
    mf.kernel()
    if not mf.converged:
        return None, None, None

    cc_obj = cc.CCSD(mf).run()
    if not cc_obj.converged:
        return None, None, None
    et = cc_obj.ccsd_t()
    e_total = float(mf.e_tot + cc_obj.e_corr + et)

    # CCSD relaxed 1-RDM in MO basis → AO basis
    dm_mo = cc_obj.make_rdm1()
    C = mf.mo_coeff
    if isinstance(dm_mo, tuple):  # UCCSD: (alpha, beta)
        dm_ao = np.array([C[0] @ dm_mo[0] @ C[0].T, C[1] @ dm_mo[1] @ C[1].T])
        n_el = float(np.einsum("ii", dm_mo[0]) + np.einsum("ii", dm_mo[1]))
    else:
        dm_ao = C @ dm_mo @ C.T
        n_el = float(np.einsum("ii", dm_mo))

    occs = natural_occupations(dm_ao, mol)
    h_vn = von_neumann_entropy(occs, n_el)
    return e_total, h_vn, n_el


def main():
    print("Cross-molecule SIRS test: does T_fit ≈ T_bond hold across G2-1?\n")

    # Compute atom data
    print("Computing atoms...")
    atom_data = {}
    for sym in ATOMS:
        amol = make_atom_mol(sym)
        e_pbe = pbe_energy(amol, uks_force=True)
        e_ccsd, h_vn, n_el = ccsd_t_with_rdm(amol)
        atom_data[sym] = {"E_PBE": e_pbe, "E_CCSDT": e_ccsd, "H_vN": h_vn, "n_el": n_el}
        print(f"  {sym}: E_PBE={e_pbe:.4f}, E_CCSDT={e_ccsd:.4f}, H_vN={h_vn:.4f}")
    print()

    # Compute molecules + atomization energies + T_fit
    print("Computing molecules and analyzing...\n")
    rows = []
    print(f"{'Mol':>5} | {'AE_PBE':>9} | {'AE_CCSDT':>9} | {'PBE err':>9} | "
          f"{'H_vN(M)':>8} | {'ΔH_vN':>8} | {'T_fit':>10} | {'T_bond':>10} | {'ratio':>6}")
    print("-" * 110)

    for name, (astr, ch, sp, count) in MOLECULES.items():
        m = make_mol(astr, ch, sp)
        e_pbe_m = pbe_energy(m)
        e_ccsd_m, h_vn_m, _n_el = ccsd_t_with_rdm(m)
        if e_pbe_m is None or e_ccsd_m is None or h_vn_m is None:
            print(f"  {name}: SKIPPED (convergence failure)")
            continue
        assert e_pbe_m is not None and e_ccsd_m is not None and h_vn_m is not None

        # Atomization energies (positive = bound)
        ae_pbe = sum(c * atom_data[s]["E_PBE"] for s, c in count.items()) - e_pbe_m
        ae_ccsd = sum(c * atom_data[s]["E_CCSDT"] for s, c in count.items()) - e_ccsd_m

        # ΔH_vN = H_vN(molecule) - sum(c × H_vN(atom)) — the molecular entropy contribution
        delta_h_vn = h_vn_m - sum(c * atom_data[s]["H_vN"] for s, c in count.items())

        # PBE error in AE
        pbe_err = ae_pbe - ae_ccsd  # positive = PBE over-binds; negative = under-binds

        # SIRS correction: AE_corrected = AE_PBE - ΔH_vN × k_B T × ln 2
        # (Sign per F = E - TS; molecular entropy contribution lowers AE if positive.)
        # We want: AE_corrected = AE_CCSDT
        # → AE_PBE - ΔH_vN × k_B T × ln 2 = AE_CCSDT
        # → T_fit = (AE_PBE - AE_CCSDT) / (ΔH_vN × k_B × ln 2)
        if abs(delta_h_vn) > 1e-6:
            T_fit = pbe_err / (delta_h_vn * K_B_HA * LN2)
        else:
            T_fit = float('nan')

        # T_bond: temperature equivalent of bond energy
        T_bond = abs(ae_ccsd) / K_B_HA  # atomization energy ÷ k_B

        ratio = T_fit / T_bond if T_bond > 0 and not math.isnan(T_fit) else float('nan')

        print(f"{name:>5} | {ae_pbe*1000:9.3f} | {ae_ccsd*1000:9.3f} | "
              f"{pbe_err*1000:+9.3f} | {h_vn_m:8.4f} | {delta_h_vn:+8.4f} | "
              f"{T_fit:10.0f} | {T_bond:10.0f} | {ratio:6.3f}")

        rows.append({
            "molecule": name,
            "AE_PBE_mHa": ae_pbe * 1000,
            "AE_CCSDT_mHa": ae_ccsd * 1000,
            "PBE_err_mHa": pbe_err * 1000,
            "H_vN_M": h_vn_m,
            "delta_H_vN": delta_h_vn,
            "T_fit_K": T_fit,
            "T_bond_K": T_bond,
            "ratio_T_fit_T_bond": ratio,
        })

    # Save
    if rows:
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Analysis
    valid = [r for r in rows if not math.isnan(r["ratio_T_fit_T_bond"]) and not math.isinf(r["ratio_T_fit_T_bond"])]
    if not valid:
        print("\nNo valid rows for analysis.")
        return

    ratios = np.array([r["ratio_T_fit_T_bond"] for r in valid])
    T_fits = np.array([r["T_fit_K"] for r in valid])
    T_bonds = np.array([r["T_bond_K"] for r in valid])

    print(f"\n{'='*72}")
    print("ANALYSIS")
    print(f"{'='*72}")
    print(f"\n  N molecules with valid T_fit:  {len(valid)}")
    print(f"  Mean   T_fit/T_bond ratio:    {ratios.mean():.3f}")
    print(f"  Median T_fit/T_bond ratio:    {np.median(ratios):.3f}")
    print(f"  Std    T_fit/T_bond ratio:    {ratios.std():.3f}")
    print(f"  Range  T_fit/T_bond ratio:    [{ratios.min():.3f}, {ratios.max():.3f}]")

    # Linear fit T_fit = a × T_bond + b
    if len(T_bonds) >= 3:
        from scipy import stats
        result = stats.linregress(T_bonds, T_fits)
        slope = float(result.slope)  # type: ignore[attr-defined]
        intercept = float(result.intercept)  # type: ignore[attr-defined]
        r2 = float(result.rvalue) ** 2  # type: ignore[attr-defined]
        print(f"\n  Linear fit T_fit = a × T_bond + b:")
        print(f"    slope a = {slope:.4f}")
        print(f"    intercept b = {intercept:.0f} K")
        print(f"    R² = {r2:.4f}")

    print(f"\n  Verdict:")
    if ratios.std() / abs(ratios.mean()) < 0.2:
        print("  → T_fit/T_bond is approximately constant across molecules.")
        print("  → SIRS chemistry has a real prediction: T = (constant) × T_bond.")
    elif ratios.std() / abs(ratios.mean()) < 0.5:
        print("  → T_fit varies substantially but correlates with T_bond.")
        print("  → SIRS chemistry has a soft prediction; needs more careful operator.")
    else:
        print("  → T_fit varies arbitrarily across molecules.")
        print("  → SIRS chemistry as currently formulated is just a per-system fit.")

    print(f"\nResults: {OUT_CSV.name}")


if __name__ == "__main__":
    main()
