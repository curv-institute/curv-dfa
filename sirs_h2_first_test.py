#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy", "scipy"]
# ///
"""
First SIRS-grounded test: H2 dissociation with von Neumann entropy correction.

After today's audit found that 4 of 4 cross-domain MLR validations failed
(see wiki/topics/empirical-validation-audit.md), the strategic pivot is to
re-derive K, H, E from SIRS λ = 1/(k_B T ln 2) with explicit dimensions.

This is the simplest possible chemistry test:

  Operator:    E_correction(R) = H_vN(R) × k_B T_eff × ln 2   [Hartree]
  H_vN(R):     von Neumann entropy of 1-RDM = -Σ n_i ln(n_i)  [nats]
  Prediction:  H_vN(R=eq) ≈ 0, H_vN(R→∞) → ln 2 (one electron on each H)
  Direction:   E_correction should preferentially raise the (incorrectly low)
               PBE energy at large R, pushing it toward CCSD(T)/FCI.

The honest test:
  - Does H_vN actually scale as predicted (0 → ln 2)?
  - For what T, if any, does (PBE + correction) match CCSD(T) on H2?
  - Is that T physically interpretable, or just a free fit parameter?

If T has to be fit, the SIRS reformulation is no better than the previous
ad-hoc operationalizations. If T is constrained to a physical value (e.g.,
room T, calculation precision T) and the correction works without tuning,
this is real progress.

Reference:
  - SIRS reformulation scoping: wiki/topics/sirs-reformulation-scoping.md
  - Empirical audit: wiki/topics/empirical-validation-audit.md
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

import numpy as np
from pyscf import cc, dft, gto, scf

OUT_DIR = Path(__file__).parent
OUT_CSV = OUT_DIR / "sirs_h2_results.csv"

# Boltzmann constant in Hartree / Kelvin
K_B_HA = 3.166811563e-6  # Ha/K
LN2 = math.log(2)

# Bond distances spanning equilibrium → long-bond multireference regime
DISTANCES = [0.6, 0.74, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Three candidate T_eff values to probe:
#   T_resolution: from typical convergence threshold ~10^-6 Ha
#   T_room:       physical room temperature
#   T_electronic: typical finite-T DFT smearing
T_CANDIDATES = {
    "T_resolution_0.3K": 0.3,        # ≈ 10^-6 Ha resolution
    "T_room_298K":       298.0,
    "T_electronic_10000K": 10000.0,
}


def run_pbe_h2(R: float) -> tuple[float, np.ndarray]:
    """RKS PBE on H2 at bond length R. Return (E_total, 1-RDM in MO basis)."""
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {R}", basis="def2-svp",
                charge=0, spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "GGA_X_PBE,GGA_C_PBE"
    mf.grids.level = 4
    mf.diis_space = 12
    mf.max_cycle = 200
    mf.level_shift = 0.1
    mf.damp = 0.2
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"PBE did not converge at R={R}")
    # 1-RDM in AO basis
    dm_ao = mf.make_rdm1()
    # Convert to MO basis (diagonal occupations would just give RHF occupations
    # which are 2,2,0,0,...; for closed-shell single-determinant DFT that gives
    # H_vN ≈ 0 trivially. We need the natural-orbital decomposition of the
    # CORRELATED density matrix to capture multireference character.)
    return float(mf.e_tot), dm_ao


def ccsd_t_h2(R: float) -> tuple[float, np.ndarray]:
    """CCSD(T) on H2 at bond length R. Return (E_total, CCSD 1-RDM in AO basis)."""
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {R}", basis="def2-svp",
                charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    cc_obj = cc.CCSD(mf).run()
    et = cc_obj.ccsd_t()
    e_total = float(mf.e_tot + cc_obj.e_corr + et)
    # CCSD relaxed density matrix in MO basis
    dm_mo = cc_obj.make_rdm1()
    # Transform to AO
    C = mf.mo_coeff
    dm_ao = C @ dm_mo @ C.T
    return e_total, dm_ao


def natural_occupations(dm_ao: np.ndarray, mol) -> np.ndarray:
    """Diagonalize 1-RDM in an orthonormal basis to get natural occupations.

    We need γ in an orthonormal basis to diagonalize. Using S^(1/2) γ S^(1/2)
    via the AO overlap matrix.
    """
    s = mol.intor("int1e_ovlp")
    # Lowdin orthogonalization
    s_eig, s_vec = np.linalg.eigh(s)
    s_half = s_vec @ np.diag(np.sqrt(np.maximum(s_eig, 0))) @ s_vec.T
    gamma_orth = s_half @ dm_ao @ s_half
    occs = np.linalg.eigvalsh(gamma_orth)
    # For closed-shell, total electrons = sum of occs ≈ N. Sort descending.
    return np.sort(occs)[::-1]


def von_neumann_entropy(occs: np.ndarray, n_electrons: int = 2) -> float:
    """S_vN = -Σ p_i ln p_i where p_i = occ_i / N (normalized to probability).

    For a single-determinant closed-shell (RHF/RKS) state, occs are 2, 2, ..., 0
    so p_i ∈ {1/N, 1/N, ..., 0} for doubly-occupied orbitals. Properly:
    - For closed-shell HF/DFT, occs/2 ∈ {1, 1, ..., 0}, so each doubly-occupied
      orbital contributes 0 to entropy of γ/N. von Neumann entropy of a single
      determinant is exactly 0.
    - For correlated (CCSD) γ, occs deviate from {2,2,...,0,0}, and S > 0.
    """
    p = occs / float(n_electrons)
    p = p[p > 1e-12]
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def main():
    print("SIRS-grounded H2 dissociation test")
    print(f"{'='*72}\n")

    rows = []
    for R in DISTANCES:
        print(f"R = {R:.2f} A")

        # PBE
        e_pbe, dm_pbe_ao = run_pbe_h2(R)
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {R}", basis="def2-svp",
                    charge=0, spin=0, verbose=0)
        occs_pbe = natural_occupations(dm_pbe_ao, mol)
        H_vN_pbe = von_neumann_entropy(occs_pbe)

        # CCSD(T)
        e_ccsdt, dm_ccsd_ao = ccsd_t_h2(R)
        occs_ccsd = natural_occupations(dm_ccsd_ao, mol)
        H_vN_ccsd = von_neumann_entropy(occs_ccsd)

        # PBE error vs CCSD(T)
        delta_pbe = e_pbe - e_ccsdt  # negative if PBE underbinds, positive if overbinds

        # Predicted SIRS correction at each candidate T
        # E_correction = H_vN(reference) × k_B T × ln 2
        # We use H_vN from CCSD(T) since that's the "true" multireference signature
        corrections = {}
        for label, T in T_CANDIDATES.items():
            corr = H_vN_ccsd * K_B_HA * T * LN2
            corrections[label] = corr

        print(f"  PBE:    E = {e_pbe:.6f} Ha,  occs = {occs_pbe[:4]},  H_vN = {H_vN_pbe:.4f}")
        print(f"  CCSD(T):E = {e_ccsdt:.6f} Ha,  occs = {occs_ccsd[:4]},  H_vN = {H_vN_ccsd:.4f}")
        print(f"  PBE - CCSD(T): {delta_pbe*1000:+.3f} mHa")
        for label, corr in corrections.items():
            print(f"  Correction at {label}: {corr*1000:+.4f} mHa")
        print()

        rows.append({
            "R_A": R,
            "E_PBE_Ha": e_pbe,
            "E_CCSDT_Ha": e_ccsdt,
            "PBE_minus_CCSDT_mHa": delta_pbe * 1000,
            "occs_PBE": ",".join(f"{x:.4f}" for x in occs_pbe[:4]),
            "occs_CCSD": ",".join(f"{x:.4f}" for x in occs_ccsd[:4]),
            "H_vN_PBE_nats": H_vN_pbe,
            "H_vN_CCSD_nats": H_vN_ccsd,
            **{f"corr_{k}_mHa": v * 1000 for k, v in corrections.items()},
        })

    # Save
    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Analysis
    print(f"{'='*72}")
    print("ANALYSIS")
    print(f"{'='*72}\n")

    print("Question 1: Does H_vN scale as predicted (0 at eq → ln 2 at dissoc)?")
    print(f"  H_vN(CCSD) at R=0.74 (eq): {rows[1]['H_vN_CCSD_nats']:.4f} nats")
    print(f"  H_vN(CCSD) at R=5.00 (∞):  {rows[-1]['H_vN_CCSD_nats']:.4f} nats")
    print(f"  Predicted long-bond limit: ln 2 = {LN2:.4f} nats")
    h_far = rows[-1]['H_vN_CCSD_nats']
    print(f"  Match to ln 2: {h_far / LN2 * 100:.1f}%")
    print()

    print("Question 2: At each candidate T, does (PBE + correction) approach CCSD(T)?")
    print(f"  PBE error grows from {rows[1]['PBE_minus_CCSDT_mHa']:+.2f} mHa (eq) "
          f"to {rows[-1]['PBE_minus_CCSDT_mHa']:+.2f} mHa (R=5)")
    print()
    for T_label in T_CANDIDATES:
        corr_far = rows[-1][f"corr_{T_label}_mHa"]
        residual_far = rows[-1]['PBE_minus_CCSDT_mHa'] + corr_far
        print(f"  At {T_label}: corr at R=5 = {corr_far:+.4f} mHa, "
              f"residual = {residual_far:+.3f} mHa")
    print()

    # Question 3: What T, if any, would make the correction match the PBE error?
    print("Question 3: What T_fit would make correction match PBE error at R=5?")
    pbe_err_far = rows[-1]['PBE_minus_CCSDT_mHa']  # mHa
    if h_far > 1e-6 and abs(pbe_err_far) > 1e-6:
        # corr (Ha) = H × k_B T × ln2 → T = -delta_pbe / (H × k_B × ln2)
        # We want correction to ADD to PBE to match CCSD(T):
        #   E_PBE + corr ≈ E_CCSDT  →  corr ≈ -delta_pbe
        T_fit = -pbe_err_far * 1e-3 / (h_far * K_B_HA * LN2)
        print(f"  T_fit = {T_fit:.0f} K")
        print(f"  k_B T_fit = {K_B_HA * T_fit * 1000:.4f} mHa = "
              f"{K_B_HA * T_fit * 27.211 * 1000:.2f} meV")
        print()
        print("  Is T_fit physically interpretable, or just a free parameter?")
        if 250 < T_fit < 350:
            print("  → Suspiciously close to room T (298 K). Possible physical T.")
        elif 1000 < T_fit < 20000:
            print("  → In the range of typical electronic temperatures (Mermin smearing).")
        elif T_fit < 1:
            print("  → Below typical numerical precision T (suggests SIRS framing too weak).")
        else:
            print("  → Doesn't match any obvious physical T scale. Likely a fit parameter.")
    print()

    print(f"Results CSV: {OUT_CSV.name}")
    print(f"\nVerdict: see wiki/topics/sirs-reformulation-scoping.md for interpretation.")


if __name__ == "__main__":
    main()
