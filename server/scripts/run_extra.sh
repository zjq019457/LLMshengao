#!/bin/bash
# Extra main-table baselines: DeepReviewer-14B (fast) and OpenReviewer-8B on dev / test25 / test24 / K1, each also gated.
# Usage: bash run_extra.sh <cfg...>   e.g. openreviewer-8b deepreviewer-14b:fast
# Each model waits for models/<dir>/UPLOAD_COMPLETE. Gates, judge and tables wait for run_paper.sh to finish
# ("paper stage done"), so the audit-check caches it builds are reused instead of written twice.
R=/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0
PY=$R/.venv/bin/python
cd $R/scripts
NEED_MIB=${NEED_MIB:-34000}
CFGS=("$@")
declare -A DIR=([openreviewer-8b]=Llama-OpenReviewer-8B [deepreviewer-14b:fast]=DeepReviewer-14B
                [cyclereviewer-8b]=CycleReviewer-ML-Llama-3.1-8B)

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
    echo "[$(date +%T)] extra GPU$gpu: $* (try $try)" >> $R/logs/k0.log
    CUDA_VISIBLE_DEVICES=$gpu $PY "$@" > $R/logs/$log.log 2>&1
    local rc=$?
    echo "[$(date +%T)] extra exit $rc : $*" >> $R/logs/k0.log
    [ $rc -eq 0 ] && return 0
    grep -qE "Free memory on device|Engine core initialization failed|CUDA out of memory" $R/logs/$log.log || return $rc
    sleep 300
  done
}

VAR=("--k 1" "--k 2" "--k 3" "--k 1 --mode minus" "--k 2 --mode minus" "--k 3 --mode minus")
SPLITS=("_dev bench_dev25 O" "_bench bench_iclr25 O" "_bench24 bench_iclr24 O" " versions NF")
# the clean ICLR 2026 split, once it has been uploaded and its audit exists (run_iclr26.sh)
[ -s $R/data/bench_iclr26.jsonl ] && SPLITS+=("_bench26 bench_iclr26 O")

for cfg in "${CFGS[@]}"; do
  until [ -f $R/models/${DIR[$cfg]}/UPLOAD_COMPLETE ]; do sleep 300; done
  for s in "${SPLITS[@]}"; do
    set -- $s; [ $# -eq 2 ] && set -- "" "$@"
    [ -s $R/out$1/reviews_${cfg/:/_}.jsonl ] && continue
    run rev$1_${cfg//[:+]/_} review.py --model $cfg --versions $3 --allow-unchecked --data $2 --tag "$1"
  done
done

until grep -q "paper stage done" $R/logs/k0.log; do sleep 300; done
# the ICLR 2026 gates need that split's audit and check cache (run_iclr26.sh)
[ -s $R/data/bench_iclr26.jsonl ] && until grep -q "iclr26 stage done" $R/logs/k0.log; do sleep 300; done
for cfg in "${CFGS[@]}"; do
  for s in "${SPLITS[@]}"; do
    set -- $s; [ $# -eq 2 ] && set -- "" "$@"
    for v in "${VAR[@]}"; do
      run gate$1_${cfg//[:+]/_}_$(echo "$v" | tr -d ' -') gate.py --base $cfg --data $2 --tag "$1" $v
    done
  done
done
for s in "_dev bench_dev25" "_bench bench_iclr25" "_bench24 bench_iclr24" "_bench26 bench_iclr26"; do
  set -- $s; [ -s $R/data/$2.jsonl ] && $PY bench_eval.py --tag $1 --data $2 > $R/logs/eval$1.log 2>&1
done
[ -s $R/data/bench_iclr26.jsonl ] && $PY cmp26.py > $R/logs/cmp26.log 2>&1
(export NEED_MIB=28000; K0_GPU_MEM=0.33 run judge_extra judge.py --judge qwen3-32b-awq)
$PY analyze.py > $R/out/k0_result.txt 2>&1
$PY paper_tables.py > $R/logs/paper_tables.log 2>&1
mkdir -p $R/out/paper && cp $R/paper/* $R/out/paper/
echo "[$(date +%T)] extra stage done" >> $R/logs/k0.log
