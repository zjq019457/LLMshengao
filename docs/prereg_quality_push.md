# Pre-registration: quality push (2026-10-29, ~08:30 PDT)

Committed after launching, and before any result of, OpenReviewer-8B with 16 samples (`openreviewer-8b:n16`) and the
AC meta-review (`qwen3-32b-awq:ac`, `meta.py`). At commit time none of these output files existed on any split.

- **Reference:** OpenReviewer-8B, 3 samples (`openreviewer-8b`). This is the strongest single reviewer on both test years.
- **Primary test:** paired paper-bootstrap (2000 resamples) of ΔAUC and ΔSpearman versus the reference, on the
  **pooled test set** (`bench_iclr25` + `bench_iclr24`, 400 papers). A system improves quality if the 95% CI of
  ΔAUC excludes 0 and ΔSpearman is not significantly negative.
- **Secondary:** the same deltas per test year and on dev. These are reported, but no claim rests on a single year.
- **Systems tested:** `openreviewer-8b:n16`; `openreviewer-8b:n16` with the gate (k2 cap, k1 minus2);
  `openreviewer-8b+gate_k2`; `qwen3-32b-awq:ac`; avg3 (unweighted z-mean of OpenReviewer, DeepReviewer-fast and
  AI Scientist).
- **No fitting:** none of these systems has parameters fit on any labelled data.
- **Reporting:** all systems are reported, including ones that do not improve. Code: `scripts/quality_cmp.py`.
