#!/usr/bin/env uv run
# /// script
# dependencies = ["pyscf", "numpy"]
# ///
"""
RIFT-DFT: Representational Information Field Theory corrections for DFT.

CLI entrypoint for running RIFT experiments.

Usage:
    uv run dft.py              # Run default test suite
    uv run dft.py --validate   # Run validation only
    uv run dft.py --scc        # Run SCC sign test
    uv run dft.py --compose    # Run composability test
"""
from __future__ import annotations

import sys

# Import experiments
from experiments import (
    run_validation,
    run_h2plus_sie_validation,
    run_h2plus_scc_test,
    run_composability_test,
    run_baseline_comparison,
    run_harmonizer_comparison,
    run_density_diagnostics,
    run_ecp_comparison,
    run_co_test,
    run_no_test,
    run_rho_s_sweep,
)


def main():
    """CLI entrypoint."""
    print("RIFT-DFT Analysis Suite")
    print("=" * 80)

    args = set(sys.argv[1:])

    # Parse simple CLI flags
    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    if "--validate" in args:
        run_validation()
        return

    if "--scc" in args:
        if not run_validation():
            print("\nAborting: baseline mismatch")
            sys.exit(1)
        run_h2plus_scc_test()
        return

    if "--compose" in args:
        if not run_validation():
            print("\nAborting: baseline mismatch")
            sys.exit(1)
        run_composability_test()
        return

    if "--baseline" in args:
        run_baseline_comparison()
        return

    if "--harmonizer" in args:
        run_harmonizer_comparison()
        return

    if "--density" in args:
        run_density_diagnostics()
        return

    if "--ecp" in args:
        run_ecp_comparison()
        return

    if "--co" in args:
        run_co_test()
        return

    if "--no" in args:
        run_no_test()
        return

    if "--sweep" in args:
        run_rho_s_sweep()
        return

    if "--all" in args:
        if not run_validation():
            print("\nAborting: baseline mismatch")
            sys.exit(1)
        run_h2plus_sie_validation()
        run_h2plus_scc_test()
        run_composability_test()
        return

    # Default: validation + SIE harness + SCC test
    if not run_validation():
        print("\nAborting: baseline mismatch needs to be fixed first")
        sys.exit(1)

    run_h2plus_sie_validation()
    run_h2plus_scc_test()


if __name__ == "__main__":
    main()
