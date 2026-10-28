#!/bin/bash
# Main-table run (continuation of `run_k0.sh paper` with retries): dev split for choosing gate settings,
# ICLR 2024 as a second test year, gate variants + ablations, judge, tables/figures.
# Usage: bash run_paper.sh [pid-to-wait-for ...]   (reviews already running from the first attempt)
R=/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0
PY=$R/.venv/bin/python
cd $R/scripts
NEED_MIB=${NEED_MIB:-34000}

freest_gpu() {  # optional arg: GPU index to skip. Blocks until a GPU has NEED_MIB free.
  while :; do
    line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | grep -v "^${1:-x}," | sort -t, -k2 -nr | head -1)
    [ "${line#*, }" -ge "$NEED_MIB" ] && { echo "${line%%,*}"; return; }
    sleep 120
  done
}

run() {  # run <log> <args...>: picks a GPU, retries while the engine cannot get its memory (GPUs are shared)
  local log=$1; shift
  for try in $(seq 1 40); do
    local gpu=$(freest_gpu)
    echo "[$(date +%T)] GPU$gpu: $* (try $try)" >> $R/logs/k0.log
    CUDA_VISIBLE_DEVICES=$gpu $PY "$@" > $R/logs/$log.log 2>&1
    local rc=$?
    echo "[$(date +%T)] exit $rc : $*" >> $R/logs/k0.log
    [ $rc -eq 0 ] && return 0
    grep -qE "Free memory on device|Engine core initialization failed|CUDA out of memory" $R/logs/$log.log || return $rc
    sleep 300
  done
}

rv() {  # rv <split-tag> <data> <cfg...>
  local t=$1 d=$2; shift 2
  for cfg in "$@"; do
    [ -s $R/out$t/reviews_${cfg/:/_}.jsonl ] && continue
    run rev${t}_${cfg//[:+]/_} review.py --model $cfg --versions O --allow-unchecked --data $d --tag "$t"
  done
}

gt() {  # gt <split-tag> <data> <base> <gate args...>
  local t=$1 d=$2 b=$3; shift 3
  run gate${t}_${b//[:+]/_}_$(echo "$*" | tr -d ' -') gate.py --base $b --data $d --tag "$t" "$@"
}

VAR=("--k 1" "--k 2" "--k 3" "--k 1 --mode minus" "--k 2 --mode minus" "--k 3 --mode minus")
ABL=("--k 1 --no-quote" "--k 1 --no-reported" "--k 1 --audit-samples 1" "--k 1 --checker qwen3-8b"
     "--k 2 --no-quote" "--k 2 --no-reported" "--k 2 --checker qwen3-8b"
     "--k 1 --audit qwen3-8b:think+audit" "--k 2 --audit qwen3-8b:think+audit")
M1=qwen3-32b-awq:think+verify+grule+s10

for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 60; done; done &
WAITER=$!

# phase 1: remaining reviews / audits (two lanes) + check caches
{ rv _bench24 bench_iclr24 deepreviewer-7b:fast qwen3-32b-awq:s10 qwen3-32b-awq:ais $M1
  rv _dev bench_dev25 qwen3-32b-awq:think+verify+s10 $M1; } &
sleep 180
{ rv _bench bench_iclr25 qwen3-8b:think+audit
  gt "" versions deepreviewer-7b:fast --k 1
  gt _bench bench_iclr25 deepreviewer-7b:fast --k 1
  gt "" versions deepreviewer-7b:fast --k 1 --checker qwen3-8b
  gt _bench bench_iclr25 deepreviewer-7b:fast --k 1 --checker qwen3-8b; } &
wait

# phase 2: gate variants (checks are cached after the first call per split/checker)
for b in deepreviewer-7b:fast qwen3-32b-awq:s10 qwen3-32b-awq:ais qwen3-32b-awq:think+verify+s10; do
  for v in "${VAR[@]}"; do gt _dev bench_dev25 $b $v; done
done
for b in deepreviewer-7b:fast qwen3-32b-awq:s10 qwen3-32b-awq:ais; do
  for v in "${VAR[@]}"; do gt _bench24 bench_iclr24 $b $v; done
done
for b in deepreviewer-7b:fast qwen3-32b-awq:s10 qwen3-32b-awq:ais deepreviewer-7b:std qwen3-32b-awq:think+verify+s10; do
  for v in "${VAR[@]}"; do gt _bench bench_iclr25 $b $v; done
done
for b in deepreviewer-7b:fast qwen3-32b-awq:ais qwen3-32b-awq:think+verify+s10 qwen3-32b-awq:think+two qwen3-8b:think+two; do
  for v in "${VAR[@]}"; do gt "" versions $b $v; done
done
for b in deepreviewer-7b:fast qwen3-32b-awq:s10; do
  for v in "${ABL[@]}"; do gt _bench bench_iclr25 $b $v; done
done
for b in deepreviewer-7b:fast qwen3-32b-awq:ais; do
  for v in "${ABL[@]}"; do gt "" versions $b $v; done
done

# phase 3: evaluation
for s in "_dev bench_dev25" "_bench bench_iclr25" "_bench24 bench_iclr24"; do
  set -- $s; $PY bench_eval.py --tag $1 --data $2 > $R/logs/eval$1.log 2>&1
done
(export NEED_MIB=28000; K0_GPU_MEM=0.33 run judge judge.py --judge qwen3-32b-awq)
$PY analyze.py > $R/out/k0_result.txt 2>&1
$PY paper_tables.py > $R/logs/paper_tables.log 2>&1
mkdir -p $R/out/paper && cp $R/paper/* $R/out/paper/   # out/ is in the backup set
echo "[$(date +%T)] paper stage done" >> $R/logs/k0.log
