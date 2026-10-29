#!/bin/bash
# OpenReviewer n16 on test24, pinned to GPU0 in parallel with run_quality.sh (which skips it via a placeholder file).
R=/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0
PY=$R/.venv/bin/python
cd $R/scripts
F=$R/out_bench24/reviews_openreviewer-8b_n16.jsonl
[ -s $F ] || echo '{"placeholder": 1}' > $F
for try in $(seq 1 40); do
  echo "[$(date +%T)] quality GPU0: or n16 bench24 (try $try)" >> $R/logs/k0.log
  CUDA_VISIBLE_DEVICES=0 $PY review.py --model openreviewer-8b:n16 --n 16 --versions O --allow-unchecked \
    --data bench_iclr24 --tag _bench24 > $R/logs/rev_bench24_or_n16.log 2>&1 && break
  grep -qE "Free memory on device|Engine core initialization failed|CUDA out of memory" $R/logs/rev_bench24_or_n16.log || exit 1
  sleep 300
done
echo "[$(date +%T)] quality GPU0: or n16 bench24 done" >> $R/logs/k0.log
for v in "--k 1" "--k 2" "--k 1 --mode minus" "--k 2 --mode minus"; do
  $PY gate.py --base openreviewer-8b:n16 --data bench_iclr24 --tag _bench24 $v >> $R/logs/gate_or_n16.log 2>&1
done
until grep -q "quality stage done" $R/logs/k0.log; do sleep 120; done
$PY bench_eval.py --tag _bench24 --data bench_iclr24 > $R/logs/eval_bench24.log 2>&1
$PY quality_cmp.py > $R/logs/quality_cmp.log 2>&1
cp $R/paper/* $R/out/paper/
echo "[$(date +%T)] quality stage done (with GPU0 test24)" >> $R/logs/k0.log
