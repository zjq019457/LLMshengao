#!/bin/bash
# Quality push: OpenReviewer with 16 samples (self-consistency) and AC meta-review, on dev / test25 / test24.
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
    echo "[$(date +%T)] quality GPU$gpu: $* (try $try)" >> $R/logs/k0.log
    CUDA_VISIBLE_DEVICES=$gpu $PY "$@" > $R/logs/$log.log 2>&1
    local rc=$?
    echo "[$(date +%T)] quality exit $rc : $*" >> $R/logs/k0.log
    [ $rc -eq 0 ] && return 0
    grep -qE "Free memory on device|Engine core initialization failed|CUDA out of memory" $R/logs/$log.log || return $rc
    sleep 300
  done
}

SPLITS=("_dev bench_dev25" "_bench bench_iclr25" "_bench24 bench_iclr24")
{ for s in "${SPLITS[@]}"; do set -- $s
    [ -s $R/out$1/reviews_openreviewer-8b_n16.jsonl ] || \
      run rev$1_or_n16 review.py --model openreviewer-8b:n16 --n 16 --versions O --allow-unchecked --data $2 --tag $1
    for v in "--k 1" "--k 2" "--k 1 --mode minus" "--k 2 --mode minus"; do
      $PY gate.py --base openreviewer-8b:n16 --data $2 --tag $1 $v >> $R/logs/gate_or_n16.log 2>&1
    done
  done; } &
sleep 180
for s in "${SPLITS[@]}"; do set -- $s
  [ -s $R/out$1/reviews_qwen3-32b-awq_ac.jsonl ] || run meta$1 meta.py --data $2 --tag $1
done
wait
for s in "${SPLITS[@]}"; do set -- $s; $PY bench_eval.py --tag $1 --data $2 > $R/logs/eval$1.log 2>&1; done
$PY quality_cmp.py > $R/logs/quality_cmp.log 2>&1
mkdir -p $R/out/paper && cp $R/paper/* $R/out/paper/
echo "[$(date +%T)] quality stage done" >> $R/logs/k0.log
