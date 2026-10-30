#!/bin/bash
# Missing experiments for the main system:
#  lane A: OpenReviewer-8B n16 on K1 (fatal-flaw versions) + gates, then the judge
#  lane B: DeepReviewer-7B fast n16 on dev / test25 / test24 + gate_k2
#  then: bench_eval, judge, analyze, paper_tables, curve.py (sample-count curve + score shift)
R=/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0
PY=$R/.venv/bin/python
cd $R/scripts
NEED_MIB=${NEED_MIB:-34000}

freest_gpu() {
  while :; do
    line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
    [ "${line#*, }" -ge "$NEED_MIB" ] && { echo "${line%%,*}"; return; }
    sleep 120
  done
}

run() {  # run <log> <args...>: retries while the engine cannot get its memory (GPUs are shared)
  local log=$1; shift
  for try in $(seq 1 40); do
    local gpu=$(freest_gpu)
    echo "[$(date +%T)] k1n16 GPU$gpu: $* (try $try)" >> $R/logs/k0.log
    CUDA_VISIBLE_DEVICES=$gpu $PY "$@" > $R/logs/$log.log 2>&1
    local rc=$?
    echo "[$(date +%T)] k1n16 exit $rc : $*" >> $R/logs/k0.log
    [ $rc -eq 0 ] && return 0
    grep -qE "Free memory on device|Engine core initialization failed|CUDA out of memory" $R/logs/$log.log || return $rc
    sleep 300
  done
}

{ [ -s $R/out/reviews_openreviewer-8b_n16.jsonl ] || \
    run rev_k1_or_n16 review.py --model openreviewer-8b:n16 --n 16 --versions NF --allow-unchecked --data versions --tag ""
  for v in "--k 1" "--k 2" "--k 1 --mode minus" "--k 2 --mode minus"; do
    $PY gate.py --base openreviewer-8b:n16 --data versions --tag "" $v >> $R/logs/gate_k1_or_n16.log 2>&1
  done; } &
sleep 240
for s in "_dev bench_dev25" "_bench bench_iclr25" "_bench24 bench_iclr24"; do set -- $s
  [ -s $R/out$1/reviews_deepreviewer-7b_fast+n16.jsonl ] || \
    run rev$1_dr_n16 review.py --model deepreviewer-7b:fast+n16 --n 16 --versions O --allow-unchecked --data $2 --tag $1
  $PY gate.py --base deepreviewer-7b:fast+n16 --data $2 --tag $1 --k 2 >> $R/logs/gate_dr_n16.log 2>&1
done
wait
for s in "_dev bench_dev25" "_bench bench_iclr25" "_bench24 bench_iclr24"; do
  set -- $s; $PY bench_eval.py --tag $1 --data $2 > $R/logs/eval$1.log 2>&1
done
(export NEED_MIB=28000; K0_GPU_MEM=0.33 run judge_k1n16 judge.py --judge qwen3-32b-awq)
$PY analyze.py > $R/out/k0_result.txt 2>&1
$PY paper_tables.py > $R/logs/paper_tables.log 2>&1
$PY curve.py > $R/logs/curve.log 2>&1
mkdir -p $R/out/paper && cp $R/paper/* $R/out/paper/
echo "[$(date +%T)] k1n16 stage done" >> $R/logs/k0.log
