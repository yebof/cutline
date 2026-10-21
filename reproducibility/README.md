# Reproducibility Report

This folder holds the online Reproducibility Report of CUTLINE ("CUTLINE: Verifying Real Product Pixels in
AI-Composited Listings", Yahui Lu and Yebo Feng): the compiled PDF and its LaTeX source. The report accompanies the
manuscript and its Supplementary Material. It holds every protocol detail, data table, figure and run note of the
evaluation that the manuscript and the supplement summarise.

- [`reproducibility_report.pdf`](reproducibility_report.pdf): the report (30 pages).
- `reproducibility_report.tex`: the source.

## Contents

Sections, tables and figures are numbered R*n*. "S*n*" and Roman-numbered tables refer to the Supplementary
Material, "main" to the manuscript.

| Section | Contents |
|---|---|
| R1 Pre-registration and development | The five pre-registered experiment blocks and their freeze commits; the registration record of every experiment, including the additional experiments E13–E21; post-registration fixes, analyses and decisions; open items; development sweeps and amendments. |
| R2 Completeness, soundness and adaptive attackers (E1–E3) | Encoder classes; honest listings per stratum; detection per attack class and upload profile; the adaptive attacker inside the tolerant cell. |
| R3 Baselines (E4) | The full E4 detection table (both upload profiles, every baseline), the per-pixel tolerant comparator E19 and the A10 threshold-hugging results. |
| R4 Residual surface and tolerant comparators (E13, E19, E21) | Registered predictions and outcomes of E13–E21, with the residual-surface figure: tone tables, declared retouch and generation beyond the certified footprint. |
| R5 Region coverage (E5, E14) | Region coverage and false-alarm causes per stratum, staging detection and capture misses. |
| R6 E5 hand audits | Protocol, agreement statistics and confusion matrix of the two registered hand audits. |
| R7 Verified hops (E6, E15, E20) | Platform re-encode ladders, the confirmatory E15 rerun and the E20 analysis of why 500×375 renditions end at G2. |
| R8 Displayed pixels under AS9 (E18) | What decoders display on the certified region, inside and outside the AS9 decoder family. |
| R9 Gas costs (E9, E17) | Every gas row of the contracts and the court, measured on the local EVM simulator (EDR) and on Besu. |
| R10 Consensus, beacon and fork on Besu (E10) | Besu QBFT devnet runs at n = 4 and n = 7: inclusion delay, halting, beacon unpredictability and fork tests. |
| R11 Ledger ablation (E11) | The five ledger back ends (platform database, CT-style log, witness-cosigned logs, the CUTLINE ledger) under one attack harness, including the Besu runs. |
| R12 Costs, storage and timing (E12, E16, E16b) | Compile, verify, hop and capture timings, bundle sizes and profiles. |
| R13 Worked example of a certified listing | Which MCUs are certified or declared, and where the drawn silhouette is eroded. |
| R14 System architecture | The end-to-end architecture figure. |
| R15 Ledger details | Consortium parameters, the two court paths, status and receipts, labels, privacy and costs. |
| R16 Positioning: full table | CUTLINE against the closest lines of work. |
| R17 Evaluation narrative | The full text of the evaluation that the manuscript condenses. |
| R18 Run integrity | Host, per-job memory caps, reruns and evaluation-code fixes. |
| R-A Evidence files | The generator and result files behind every figure and generated table. |

## Main results

These numbers are quoted from the report (R2, R3 and R17).

- **Completeness.** 17,940 honest listings in 60 strata produced 0 certified-block failures and 0 container
  rejections.
- **Soundness.** Six attack classes in six upload profiles were detected in all 36 cells.
- **Baselines.** The exact test detected 100% of every attack class on both upload profiles and convicted 0 of 299
  honest test listings on each.
- **Court.** The on-chain court equals the reference implementation on all 22,306 differential cases, on Besu as on
  EDR.

## Naming

The evidence tiers X0, X1, XP and tier X of the report are called E0, E1, EP and tier E in the code, the
registrations and some figures of the report.

## Building

The report compiles with [tectonic](https://tectonic-typesetting.github.io/) (XeTeX); it needs no other files:

```bash
tectonic reproducibility_report.tex
```

| File | Role |
|---|---|
| `reproducibility_report.tex` | Report source. |
| `supp_style.tex` | Layout shared with the Supplementary Material. |
| `macros.tex` | Notation and identifier macros. |
| `references.bib` | The references the report cites. |
| `figs/` | Figure PDFs. |
| `main.aux`, `supplement.aux` | Label files of the manuscript and the supplement, read by `xr-hyper` so that references such as "main Sec. VIII" resolve. |

The paths in Appendix R-A are relative to the authors' research repository. The method code they describe is the code
in this repository.
