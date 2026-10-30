#!/bin/bash
# Four follow-ups: (1) severity decomposition, (2) human vs LLM weighing gap, (3) base vs review-fine-tuned models
# on K1, (4) held-out flaw types (K2) with the main reviewers, the audit and the gates.
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
    echo "[$(date +%T)] four GPU$gpu: $* (try $try)" >> $R/logs/k0.log
    CUDA_VISIBLE_DEVICES=$gpu $PY "$@" > $R/logs/$log.log 2>&1
    local rc=$?
    echo "[$(date +%T)] four exit $rc : $*" >> $R/logs/k0.log
    [ $rc -eq 0 ] && return 0
    grep -qE "Free memory on device|Engine core initialization failed|CUDA out of memory" $R/logs/$log.log || return $rc
    sleep 300
  done
}

K2_REV=(qwen3-32b-awq:think+audit openreviewer-8b deepreviewer-7b:fast qwen3-32b-awq:ais qwen3-32b-awq:think+two
        qwen3-32b-awq:think+verify+s10 cyclereviewer-8b openreviewer-8b:n16)
BASES=(llama31-8b-instruct:orfmt llama31-8b-instruct:crfmt qwen25-7b-instruct:fast+drfmt)

# lane 1: K2 build, reviews, gates, judge
{ [ -s $R/data/versions_k2.jsonl ] || run k2_build make_versions.py k2 --model qwen3-32b-awq
  for cfg in "${K2_REV[@]}"; do
    [ -s $R/out_k2/reviews_${cfg/:/_}.jsonl ] && continue
    extra=""; [ "$cfg" = openreviewer-8b:n16 ] && extra="--n 16"
    run revk2_${cfg//[:+]/_} review.py --model $cfg --versions NF --allow-unchecked --data versions_k2 --tag _k2 $extra
  done
  for cfg in "${K2_REV[@]:1}"; do
    for k in 1 2; do run gatek2_${cfg//[:+]/_}_k$k gate.py --base $cfg --data versions_k2 --tag _k2 --k $k; done
  done
  } &
sleep 240
# lane 2: human gap, then the base models on K1 once their upload is complete
{ run human_gap human_gap.py
  for cfg in "${BASES[@]}"; do
    m=${cfg%%:*}; d=$([ $m = qwen25-7b-instruct ] && echo Qwen2.5-7B-Instruct || echo Llama-3.1-8B-Instruct)
    until [ -f /diskF/jiaqizhang/models/$d/UPLOAD_COMPLETE ]; do sleep 300; done
    [ -s $R/out/reviews_${cfg/:/_}.jsonl ] || \
      run revbase_${cfg//[:+]/_} review.py --model $cfg --versions NF --allow-unchecked --data versions --tag ""
  done; } &
wait
# after both lanes: the two judge runs share one cache file, so they never run at the same time
(export NEED_MIB=28000 K0_GPU_MEM=0.33
 run judge_k2 judge.py --judge qwen3-32b-awq --data versions_k2 --tag _k2
 run judge_base judge.py --judge qwen3-32b-awq)
$PY severity.py > $R/logs/severity.log 2>&1
$PY k2_table.py > $R/logs/k2_table.log 2>&1
mkdir -p $R/out/paper && cp $R/paper/tab_severity* $R/paper/tab_human_gap* $R/paper/tab_k2* $R/out/paper/ 2>/dev/null
echo "[$(date +%T)] four stage done" >> $R/logs/k0.log
