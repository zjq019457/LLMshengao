#!/bin/bash
# Clean ICLR 2026 test split (data/bench_iclr26.jsonl, tag _bench26): main reviewers, the audit, gates, eval, comparison.
R=/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0
PY=$R/.venv/bin/python
cd $R/scripts
NEED_MIB=${NEED_MIB:-34000}
T=_bench26
D=bench_iclr26

freest_gpu() {
  while :; do
    # GPUs listed in $R/SKIP_GPUS (e.g. "0" or "0|2") are left to other users
    line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | grep -vE "^($(cat $R/SKIP_GPUS 2>/dev/null || echo none))," | sort -t, -k2 -nr | head -1)
    [ -n "$line" ] && [ "${line#*, }" -ge "$NEED_MIB" ] && { echo "${line%%,*}"; return; }
    sleep 120
  done
}

run() {  # run <log> <args...>: retries while the engine cannot get its memory (GPUs are shared)
  local log=$1; shift
  for try in $(seq 1 40); do
    local gpu=$(freest_gpu)
    echo "[$(date +%T)] iclr26 GPU$gpu: $* (try $try)" >> $R/logs/k0.log
    CUDA_VISIBLE_DEVICES=$gpu $PY "$@" > $R/logs/$log.log 2>&1
    local rc=$?
    echo "[$(date +%T)] iclr26 exit $rc : $*" >> $R/logs/k0.log
    [ $rc -eq 0 ] && return 0
    grep -qE "Free memory on device|Engine core initialization failed|CUDA out of memory" $R/logs/$log.log || return $rc
    sleep 300
  done
}

rv() {  # rv <cfg> [extra review.py args]
  local cfg=$1; shift
  [ -s $R/out$T/reviews_${cfg/:/_}.jsonl ] && return
  run rev${T}_${cfg//[:+]/_} review.py --model $cfg --versions O --allow-unchecked --data $D --tag $T "$@"
}

{ rv qwen3-32b-awq:think+audit
  rv openreviewer-8b
  rv openreviewer-8b:n16 --n 16; } &
sleep 240
{ rv deepreviewer-7b:fast
  rv qwen3-32b-awq:ais
  rv qwen3-32b-awq:s10
  rv qwen3-32b-awq:think+verify+s10
  rv qwen3-32b-awq:think+verify+grule+s10; } &
wait
for b in openreviewer-8b openreviewer-8b:n16 deepreviewer-7b:fast qwen3-32b-awq:ais qwen3-32b-awq:s10 qwen3-32b-awq:think+verify+s10; do
  for v in "--k 1" "--k 2" "--k 1 --mode minus" "--k 2 --mode minus"; do
    run gate${T}_${b//[:+]/_}_$(echo "$v" | tr -d ' -') gate.py --base $b --data $D --tag $T $v
  done
done
$PY bench_eval.py --tag $T --data $D > $R/logs/eval$T.log 2>&1
$PY cmp26.py > $R/logs/cmp26.log 2>&1
mkdir -p $R/out/paper && cp $R/paper/tab_iclr26* $R/out/paper/
echo "[$(date +%T)] iclr26 stage done" >> $R/logs/k0.log
