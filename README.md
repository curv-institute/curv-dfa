# Curv-DFA

**Curvature-Controlled Architecture for Density Functional Approximations**

This repository contains the reference implementation for the Curv-DFA framework described in:

> J. W. Miller, "Curv-DFA: A Curvature-Controlled Architecture for Density Functional Approximations in DFT," CURV Institute, 2026.
>
> **Publication:** https://curv.institute/publications/curv-dfa/

## Overview

Curv-DFA provides a diagnostic and control layer for existing density functional approximations (DFAs) rather than proposing a universal functional. The framework addresses two distinct failure mechanisms in DFT through independent correction channels:

1. **Fisher + Harmonizer Channel** — XC-side correction for density geometry and bonding control using Fisher information-based regularization
2. **Self-Coupling Cancellation (SCC)** — Hartree-side correction for self-interaction error in one-electron-dominated regions

A core finding is that self-interaction error and density geometry distortion are distinct failures requiring separate control mechanisms that compose stably.

## Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- PySCF
- NumPy

## Usage

All scripts use [PEP 723](https://peps.python.org/pep-0723/) inline dependencies and can be run directly with `uv`:

```bash
# Run the main test suite
uv run dft.py

# Run specific experiments
uv run dft.py --validate    # Baseline validation
uv run dft.py --scc         # SCC sign-correct test on H2+
uv run dft.py --compose     # Composability test (Fisher + SCC)
uv run dft.py --all         # Run all experiments
```

### Available Flags

| Flag | Description |
|------|-------------|
| `--validate` | Run baseline validation against known values |
| `--scc` | Run SCC sign-correct test on H2+ |
| `--compose` | Run composability test for channel independence |
| `--baseline` | Compare baseline DFT methods |
| `--harmonizer` | Test harmonizer gate effects |
| `--density` | Run density diagnostics |
| `--all` | Run complete test suite |

## Key Modules

### `rift_functionals.py`

Core implementation of the correction channels:

- `FisherModule` / `FisherCfg` — Fisher + Harmonizer XC correction
- `SCCModule` / `SCCCfg` — Self-Coupling Cancellation for Hartree term

```python
from rift_functionals import FisherModule, SCCModule, FisherCfg, SCCCfg
from pyscf import gto, dft

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pVTZ')
mf = dft.UKS(mol)
mf.xc = "GGA_X_PBE,GGA_C_PBE"

# Attach corrections
fisher = FisherModule(FisherCfg(alpha=0.01))
fisher.attach(mf)

scc = SCCModule(SCCCfg(lam=0.1))
scc.attach(mf)

mf.kernel()
```

### `experiments.py`

Numerical experiments validating the framework:

- Baseline comparisons (PBE, B3LYP, HF)
- H2+ self-interaction error tests
- Composability analysis demonstrating channel independence
- Sign-correct tests with positive controls

### `composability_test.py`

Standalone composability analysis for H2 and H2+ systems, testing whether Fisher and SCC channels interfere or compose additively.

## Results

Pre-computed results from composability tests are included:

- `composability_h2.csv` — H2 molecule results
- `composability_h2plus.csv` — H2+ radical cation results

## Citation

```bibtex
@article{miller2026curvdfa,
  title={Curv-DFA: A Curvature-Controlled Architecture for Density Functional Approximations in DFT},
  author={Miller, J. W.},
  institution={CURV Institute},
  year={2026},
  url={https://curv.institute/publications/curv-dfa/}
}
```

## License

Copyright 2026 CURV Institute. All rights reserved.

## Contact

CURV Institute, Geneva, Switzerland
https://curv.institute
