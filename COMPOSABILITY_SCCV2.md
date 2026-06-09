# Composability Regeneration with SCC v2 (2026-06-09)

## Why

`MLR_VS_FISHER_FINDINGS.md` (Honest caveats §3) flagged that the original
`composability_h2.csv` / `composability_h2plus.csv` were generated with the
retired inline SCC v1 and "should be regenerated with v2 if they're cited
elsewhere. The 2% composability claim was tested on v1." The claim is cited in
the CURV Institute curv-dfa paper (annotated as SCC-v1 in the 2026-06-09
walkback, `curv.institute@d3a146a`). This regeneration discharges that action
item (see `curv-wiki/WALKBACK_INVENTORY.md`, S7).

## What was run

`composability_test_sccv2.py` — a minimal port of `composability_test.py` with
an unchanged protocol (H2 and H2⁺, def2-svp, PBE base, same distance grids,
same calibration procedure, same additivity metric
ε(R) = dE_FS − (dE_F + dE_S)), where the Fisher and SCC channels are taken
from the canonical current implementation in `rift_functionals.py`
(`FisherModule`, `SCCModule`). `SCCModule` is SCC v2: it adds the
density-suppression gate h(ρ) = 1/(1+(ρ/ρ_s)^p) (defaults ρ_s = 0.03,
h_power = 6, w_power = 4) that v1 lacked. PySCF 2.13.1. v1 CSVs left
untouched; v2 outputs written alongside as `*_sccv2.csv`.

## Results: v1 vs v2, same metric (max over the R grid)

| System | CSV | max \|ε\| (mHa) | max \|ε\|/(\|dE_F\|+\|dE_S\|) |
|---|---|---:|---:|
| H2  | `composability_h2.csv` (v1)        | 4.268 | 1.40% |
| H2  | `composability_h2_sccv2.csv` (v2)  | 1.505 | **1.38%** |
| H2⁺ | `composability_h2plus.csv` (v1)       | 1.113 | 1.81% |
| H2⁺ | `composability_h2plus_sccv2.csv` (v2) | 0.487 | **0.99%** |

All SCF points converged (every mode, every R). SCC remains negative at
mid-range R in H2⁺ and the combined run preserves the SCC channel's behavior.

## Verdict

**The <2% composability claim is CONFIRMED under SCC v2, and tightened.**
Max relative composability error across both systems: **1.38%** (was 1.81% on
v1 under the same metric). Absolute ε shrank ~3× on H2 and ~2× on H2⁺ — the
v2 density-suppression gate removes v1's over-correction (e.g. H2 dE_S at
R = 0.5 Å: −309 mHa under v1 → −80 mHa under v2) without breaking additivity.

Scope unchanged and explicit: this is a two-system (H2, H2⁺), single-basis
(def2-svp), single-base-functional (PBE) result about the *additivity of the
Fisher and SCC correction channels in this implementation*. It is not a
cross-domain MLR claim and does not bear on λ_E / `lambda_var`
(see the λ catalog and `curv-rift/notes/LAMBDA_EQUIVALENCE_CHECK.md`).

## Provenance

- Code: `composability_test_sccv2.py` at this commit; SCC/Fisher from
  `rift_functionals.py` at this commit.
- Environment: PySCF 2.13.1, python 3.12, 2026-06-09.
- Raw outputs: `composability_h2_sccv2.csv`, `composability_h2plus_sccv2.csv`.
