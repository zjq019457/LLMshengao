#!/bin/bash
# Evidence experiments: real-flaw validation + human alignment (Qwen3-32B judge), OpenReviewer n32 on test25 / test24
# for the cost-vs-effect comparison (n32 without gate vs n16 with gate).
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
    echo "[$(date +%T)] more GPU$gpu: $* (try $try)" >> $R/logs/k0.log
    CUDA_VISIBLE_DEVICES=$gpu $PY "$@" > $R/logs/$log.log 2>&1
    local rc=$?
    echo "[$(date +%T)] more exit $rc : $*" >> $R/logs/k0.log
    [ $rc -eq 0 ] && return 0
    grep -qE "Free memory on device|Engine core initialization failed|CUDA out of memory" $R/logs/$log.log || return $rc
    sleep 300
  done
}

run real_flaw real_flaw.py &
sleep 240
for s in "_bench bench_iclr25" "_bench24 bench_iclr24"; do set -- $s
  [ -s $R/out$1/reviews_openreviewer-8b_n32.jsonl ] || \
    run rev$1_or_n32 review.py --model openreviewer-8b:n32 --n 32 --versions O --allow-unchecked --data $2 --tag $1
done
wait
$PY cost_curve.py > $R/logs/cost_curve.log 2>&1
mkdir -p $R/out/paper && cp $R/paper/tab_real_flaw* $R/paper/tab_human_align* $R/paper/tab_cost_curve* $R/out/paper/ 2>/dev/null
echo "[$(date +%T)] more stage done" >> $R/logs/k0.log
