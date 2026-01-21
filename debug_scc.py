#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy"]
# ///
"""Debug SCC: test BINDING ENERGY shifts at multiple R values."""

from dft import attach_scc_hartree
from pyscf import gto, dft, scf
import numpy as np

def run_point(R, lam=0.0):
    """Run PBE, UHF, and SCC at bond length R."""
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {R}',
        basis='def2-svp',
        charge=1,
        spin=1,
        verbose=0,
    )

    # UHF (exact)
    mf_uhf = scf.UHF(mol)
    mf_uhf.diis_space = 12
    mf_uhf.max_cycle = 150
    if R > 2.0:
        mf_uhf.level_shift = 0.2
    mf_uhf.kernel()
    e_uhf = mf_uhf.e_tot

    # PBE
    mf_pbe = dft.UKS(mol)
    mf_pbe.grids.level = 4
    mf_pbe.xc = 'GGA_X_PBE,GGA_C_PBE'
    mf_pbe.diis_space = 12
    mf_pbe.max_cycle = 150
    if R > 2.0:
        mf_pbe.level_shift = 0.2
    mf_pbe.kernel()
    e_pbe = mf_pbe.e_tot

    # SCC
    if lam > 0:
        mf_scc = dft.UKS(mol)
        mf_scc.grids.level = 4
        mf_scc.xc = 'GGA_X_PBE,GGA_C_PBE'
        mf_scc.diis_space = 12
        mf_scc.max_cycle = 150
        if R > 2.0:
            mf_scc.level_shift = 0.2
        mf_scc.kernel()
        dm0 = mf_scc.make_rdm1()
        attach_scc_hartree(mf_scc, lam=lam, w_power=4)
        mf_scc.kernel(dm0=dm0)
        e_scc = mf_scc.e_tot
    else:
        e_scc = e_pbe

    return e_uhf, e_pbe, e_scc

print("=== H₂⁺ Binding Curve: SCC Sign Test ===")
print()
print("Testing with small λ values to check direction of SIE correction.")
print("Correct SIE: at mid-R, UHF binds more than PBE; SCC should move toward UHF.")
print()

# Scan λ values
lam_values = [0.0, 0.05, 0.1]
distances = [1.0, 1.5, 2.0, 3.0, 5.0]

# Collect results
results = {}
for lam in lam_values:
    results[lam] = []
    for R in distances:
        e_uhf, e_pbe, e_scc = run_point(R, lam)
        results[lam].append({
            'R': R,
            'uhf': e_uhf,
            'pbe': e_pbe,
            'scc': e_scc,
        })

# Print absolute energies
print("--- Absolute Energies ---")
header = f"{'R(Å)':>6}  {'UHF':>12}  {'PBE':>12}"
for lam in lam_values[1:]:
    header += f"  {'SCC λ='+str(lam):>12}"
print(header)
print("-" * len(header))

for i, R in enumerate(distances):
    row = f"{R:6.2f}  {results[0][i]['uhf']:12.8f}  {results[0][i]['pbe']:12.8f}"
    for lam in lam_values[1:]:
        row += f"  {results[lam][i]['scc']:12.8f}"
    print(row)

# Binding energies relative to R=5.0
print()
print("--- Binding Energy (E - E_∞) in mHa ---")
e_inf = {lam: results[lam][-1] for lam in lam_values}

header = f"{'R(Å)':>6}  {'UHF':>10}  {'PBE':>10}  {'Δ(UHF-PBE)':>12}"
for lam in lam_values[1:]:
    header += f"  {'SCC λ='+str(lam):>10}  {'Δ(SCC-PBE)':>12}"
print(header)
print("-" * len(header))

for i, R in enumerate(distances):
    be_uhf = (results[0][i]['uhf'] - e_inf[0]['uhf']) * 1000
    be_pbe = (results[0][i]['pbe'] - e_inf[0]['pbe']) * 1000
    d_uhf = be_uhf - be_pbe

    row = f"{R:6.2f}  {be_uhf:+10.2f}  {be_pbe:+10.2f}  {d_uhf:+12.2f}"

    for lam in lam_values[1:]:
        be_scc = (results[lam][i]['scc'] - e_inf[lam]['scc']) * 1000
        d_scc = be_scc - be_pbe
        row += f"  {be_scc:+10.2f}  {d_scc:+12.2f}"

    print(row)

print()
print("--- Interpretation ---")
print("UHF-PBE: Negative means UHF binds MORE than PBE (PBE has SIE at large R)")
print("SCC-PBE: Should match UHF-PBE sign if SCC corrects SIE")
print()

# Check sign at R=2.0 (mid-bond)
idx_2 = distances.index(2.0)
be_uhf_2 = (results[0][idx_2]['uhf'] - e_inf[0]['uhf']) * 1000
be_pbe_2 = (results[0][idx_2]['pbe'] - e_inf[0]['pbe']) * 1000
d_uhf_2 = be_uhf_2 - be_pbe_2

print(f"At R = 2.0 Å:")
print(f"  Δ(UHF-PBE) = {d_uhf_2:+.2f} mHa")
for lam in lam_values[1:]:
    be_scc_2 = (results[lam][idx_2]['scc'] - e_inf[lam]['scc']) * 1000
    d_scc_2 = be_scc_2 - be_pbe_2
    sign = "✓ SAME" if d_scc_2 * d_uhf_2 > 0 else "✗ OPPOSITE"
    print(f"  Δ(SCC λ={lam}-PBE) = {d_scc_2:+.2f} mHa ({sign})")
