#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy"]
# ///
import numpy as np
from pyscf import gto, dft
from pyscf.dft import libxc
from pyscf.dft.numint import NumInt

mol = gto.M(atom='H 0 0 0; H 0 0 1.5', basis='sto-3g', charge=1, spin=1, verbose=0)
mf = dft.UKS(mol)
mf.grids.level = 1
mf.grids.build()
mf.xc = 'PBE'

# Get a sample rho for UKS (need spin-polarized format)
ni = NumInt()
dm = mf.get_init_guess()
print(f'dm type: {type(dm)}, shape: {dm.shape if hasattr(dm, "shape") else [d.shape for d in dm]}')

ao = ni.eval_ao(mol, mf.grids.coords, deriv=1)
# For UKS, need to evaluate rho for each spin channel
if isinstance(dm, np.ndarray) and dm.ndim == 3:
    rho_a = ni.eval_rho(mol, ao, dm[0], xctype='GGA')
    rho_b = ni.eval_rho(mol, ao, dm[1], xctype='GGA')
    rho = np.stack([rho_a, rho_b])
else:
    rho = ni.eval_rho(mol, ao, dm, xctype='GGA')
print(f'rho shape: {rho.shape}')

# Evaluate XC
exc, vxc, _, _ = libxc.eval_xc('GGA_X_PBE,GGA_C_PBE', rho, spin=1, deriv=1)
print(f'exc shape: {exc.shape}')
print(f'vxc type: {type(vxc)}')
if isinstance(vxc, (tuple, list)):
    print(f'vxc len: {len(vxc)}')
    for i, v in enumerate(vxc):
        arr = np.asarray(v)
        print(f'vxc[{i}] shape: {arr.shape}, dtype: {arr.dtype}')
else:
    arr = np.asarray(vxc)
    print(f'vxc shape: {arr.shape}')
