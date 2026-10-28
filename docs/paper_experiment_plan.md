# Knowing Is Not Weighing: main-table experiment plan

Written 2026-10-28 before the dev and ICLR 2024 results existed. The gate-setting selection rule in `scripts/paper_tables.py` was committed alongside this plan, before those results.

## Two methods (both in one paper)
- **M1, fatal-flaw rule:** a prompt-level method: `think+verify+grule`. The reviewer itself must mark verified fatal flaws, and the scoring rule then forces a reject.
- **M2, hard-flaw gate:** a plug-in that works on any reviewer (`gate.py`):
  1. An independent audit (Qwen3-32B, 3 samples) lists issues.
  2. Each issue is checked in a fresh session: does the paper itself state the fact (verbatim quote, checked by string match), and is it fatal?
  3. The gate fires when at least k audit samples have a verified fatal issue. The base score is then capped at ≤3/10, or lowered by 2 points with the `minus` mode.

## Data
| Split | Use |
|---|---|
| `bench_dev25` | 180 ICLR 2025 papers (90 accept / 90 reject), disjoint from test. **Only** used to choose the gate setting. |
| `bench_iclr25` | 200 ICLR 2025 papers. Main test split for quality. |
| `bench_iclr24` | 200 ICLR 2024 papers. Second test year. |
| K1 | 32 papers × (1 clean N + 4 flaw types, 9 phrasings) = 118 manuscripts. Sensitivity. |

All benchmark splits apply the same 17k–84k character length window.

## Pre-registered selection
- **Candidates:** k1 cap, then k1 minus2, k2 cap, k2 minus2, k3 cap, k3 minus2 (most aggressive first).
- **Rule:** pick the first one whose mean change on dev (DeepReviewer-fast, Qwen3-32B, AI Scientist) is ΔAUC ≥ −0.005 and ΔSpearman ≥ −0.010.
- **Reporting:** the chosen setting goes unchanged to test25, test24 and K1. k1 cap is also reported as the "strict" variant.

## Main tables
1. **Quality (test25, test24):**
   - Rows: base, +gate (strict), +gate (chosen), and M1.
   - Columns: AUC, Spearman, MAE, accuracy, paired-bootstrap Δ with 95% CI, gate fire rate, and the share of fired papers that were accepted.
2. **K1 sensitivity:**
   - Columns: detection on F, false positives on N, P(pass | F), P(pass | N), paired F−N, all with paper-bootstrap CIs.
   - Systems: DeepReviewer, AI Scientist, Qwen3-32B think+verify, two-stage, M1, and the gated versions.

## Ablations (test25 + K1)
- No quote verification; no "reported" requirement (fatal judgment alone suffices).
- 1 audit sample instead of 3.
- Checker Qwen3-8B instead of 32B; auditor Qwen3-8B instead of 32B.
- k ∈ {1, 2, 3} × {cap, minus2} (the full sweep, also on dev).

## Overhead
- `tab_cost`: calls, input/output tokens per paper, and wall-clock GPU minutes per paper.
- Stages: each reviewer, the audit, the gate check.
- GPUs are shared with other jobs, so GPU minutes are an upper bound.

## Figures (`paper/fig_*.png` plus the `.csv` behind each)
- `fig_k1_pass`: pass rate on clean vs flawed manuscripts per system.
- `fig_k1_detect_heatmap`: detection rate per system × flaw phrasing.
- `fig_tradeoff`: dev quality change vs K1 pass | flawed, per gate setting.
- `fig_k1_scores`: score histograms N vs F, base vs gated.

## Still missing for submission (not run here)
1. **Larger or closed backbone:** a larger or closed model as reviewer and checker (for example GPT-class via Codex, or Qwen3-235B). This is the most likely reviewer request.
2. **Human validation of the judge:** at least 100 judged reviews labelled by two people. The current judge check is Claude blind relabelling.
3. **Naturally occurring flaws:** real retracted or erratum papers. K1 uses injected flaws.
4. **Injection quality audit:** for example, one stat2 edit was placed on a non-main result. Re-check all `k1_check.csv` edits and drop or fix the bad ones.
5. **Hidden-leakage phrasings (leak2/leak3):** detection is still 8–40%. Report this as a limitation or design a targeted audit prompt.
6. **Gate trigger-rate analysis on real papers:** what the fired "fatal" quotes are. This needs a qualitative table of 10–20 examples.
