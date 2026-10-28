#!/bin/bash
# K0 pipeline. Usage: bash run_k0.sh reviews_qwen | reviews_dr | judge
# GPUs are shared with StepKV: each stage picks the GPU with the most free memory.
R=/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0
PY=$R/.venv/bin/python
cd $R/scripts

NEED_MIB=${NEED_MIB:-34000}  # queue until a GPU has this much free (vLLM budget + headroom)

freest_gpu() {  # optional arg: GPU index to skip. Blocks until a GPU has NEED_MIB free.
  while :; do
    line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | grep -v "^${1:-x}," | sort -t, -k2 -nr | head -1)
    [ "${line#*, }" -ge "$NEED_MIB" ] && { echo "${line%%,*}"; return; }
    echo "[$(date +%T)] waiting for GPU (best: $line MiB free)" >> $R/logs/k0.log
    sleep 300
  done
}

run() {  # run <gpu> <log> <args...>
  local gpu=$1 log=$2; shift 2
  echo "[$(date +%T)] GPU$gpu: $*" | tee -a $R/logs/k0.log
  CUDA_VISIBLE_DEVICES=$gpu $PY "$@" > $R/logs/$log.log 2>&1
  echo "[$(date +%T)] exit $? : $*" | tee -a $R/logs/k0.log
}

case $1 in
  reviews_qwen)
    g1=$(freest_gpu); g2=$(freest_gpu $g1)
    run $g1 rev_qwen3-32b-awq review.py --model qwen3-32b-awq &
    sleep 120  # let the first engine claim its memory before choosing again
    run $g2 rev_qwen3-8b review.py --model qwen3-8b
    wait ;;
  reviews_dr)
    g=$(freest_gpu)
    run $g rev_dr_fast review.py --model deepreviewer-7b:fast
    run $g rev_dr_std review.py --model deepreviewer-7b:std ;;
  k0b)  # full K0b chain: 3 Qwen3-32B configs, then DeepReviewer once uploaded, then judge + analysis
    for cfg in think verify think+verify; do
      g=$(freest_gpu)
      run $g rev_32b_${cfg/+/_} review.py --model qwen3-32b-awq:$cfg
    done
    until [ -f $R/models/DeepReviewer-7B/UPLOAD_COMPLETE ]; do
      echo "[$(date +%T)] waiting for DeepReviewer upload" >> $R/logs/k0.log; sleep 600
    done
    bash $0 reviews_dr
    bash $0 judge ;;
  k0c_versions)  # fixed leakage template on 20 papers (K0/K0b data archived in *_k0 files and out_k0b/)
    g=$(freest_gpu)
    run $g k0c_template make_versions.py template ;;
  k0c_reviews)  # H1 only (N, F); DeepReviewer added if its upload has finished
    for cfg in "" ":think+verify" ":think+verify+rule"; do
      g=$(freest_gpu)
      run $g rev_k0c_32b${cfg//[:+]/_} review.py --model qwen3-32b-awq$cfg --versions NF --allow-unchecked
    done
    if [ -f $R/models/DeepReviewer-7B/UPLOAD_COMPLETE ]; then
      g=$(freest_gpu)
      run $g rev_k0c_dr_fast review.py --model deepreviewer-7b:fast --versions NF --allow-unchecked
    fi
    bash $0 judge ;;
  k0d)  # external baselines + other open backbones + generic-rule variant of our fix, on the K0c set
    for cfg in qwen3-32b-awq:think+verify+grule qwen3-32b-awq:ais qwen3-32b-awq:ais+think deepreviewer-7b:std \
               qwen3-8b:think+verify qwen3-8b:think+verify+grule r1-llama-8b:think+verify r1-llama-8b:think+verify+grule; do
      g=$(freest_gpu)
      run $g rev_k0d_${cfg//[:+]/_} review.py --model $cfg --versions NF --allow-unchecked
    done
    bash $0 judge ;;
  k0e)  # step 1 of the stress tests: 1-10 scale, and two-stage (audit list only -> fresh-session scoring)
    for cfg in qwen3-32b-awq:think+verify+s10 qwen3-32b-awq:think+audit qwen3-32b-awq:think+two; do
      g=$(freest_gpu)
      run $g rev_k0e_${cfg//[:+]/_} review.py --model $cfg --versions NF --allow-unchecked
    done
    bash $0 judge ;;
  k1)  # 4 flaw types x 2-3 phrasings on ~30 papers; K0 outputs archived to out_k0e/, data/*_k0c.jsonl
    [ -d $R/out_k0e ] || { mv $R/out $R/out_k0e; mkdir $R/out
      cp $R/data/versions.jsonl $R/data/versions_k0c.jsonl; cp $R/data/originals.jsonl $R/data/originals_k0c.jsonl; }
    $PY make_versions.py fetch --papers $R/scripts/papers30.txt > $R/logs/k1_fetch.log 2>&1
    g=$(freest_gpu); run $g k1_versions make_versions.py k1
    lane() {  # lane <cfg...>: runs configs one after another on whichever GPU is freest
      for cfg in "$@"; do g=$(freest_gpu); run $g rev_k1_${cfg//[:+]/_} review.py --model $cfg --versions NF --allow-unchecked; done
    }
    lane qwen3-32b-awq:think+audit qwen3-32b-awq:think+two qwen3-32b-awq:think+verify+s10 qwen3-32b-awq:think+verify+grule &
    sleep 300
    lane qwen3-8b:think+audit qwen3-8b:think+two deepreviewer-7b:fast qwen3-32b-awq:ais
    wait
    bash $0 judge ;;
  bench)  # review quality on 200 ICLR 2025 papers with human ratings (DeepReview-13K test_2025)
    mkdir -p $R/out_bench
    blane() {
      for cfg in "$@"; do g=$(freest_gpu)
        run $g bench_${cfg//[:+]/_} review.py --model $cfg --versions O --allow-unchecked --data bench_iclr25 --tag _bench
      done
    }
    blane qwen3-32b-awq:s10 qwen3-32b-awq:think+verify+grule+s10 qwen3-32b-awq:think+verify+s10 &
    sleep 300
    blane deepreviewer-7b:fast qwen3-32b-awq:ais deepreviewer-7b:std
    wait
    $PY bench_eval.py ;;
  gate)  # hard-flaw gate on top of existing reviewers: quality benchmark, then K1 sensitivity
    g=$(freest_gpu); run $g bench_audit review.py --model qwen3-32b-awq:think+audit --versions O --allow-unchecked --data bench_iclr25 --tag _bench
    for b in qwen3-32b-awq:s10 deepreviewer-7b:fast qwen3-32b-awq:ais deepreviewer-7b:std; do
      [ -f $R/out_bench/reviews_${b/:/_}.jsonl ] || continue
      g=$(freest_gpu); run $g bench_gate_${b//[:+]/_} gate.py --base $b --data bench_iclr25 --tag _bench
    done
    $PY bench_eval.py
    for b in qwen3-32b-awq:think+verify+s10 deepreviewer-7b:fast qwen3-32b-awq:ais; do
      g=$(freest_gpu); run $g k1_gate_${b//[:+]/_} gate.py --base $b
    done
    bash $0 judge ;;
  paper)  # main-table run: dev split for choosing gate settings, ICLR 2024 as a second test year, gate variants + ablations
    $PY make_bench.py > $R/logs/make_bench.log 2>&1
    mkdir -p $R/out_dev $R/out_bench24 $R/out_gate_old
    mv $R/out/*+gate.jsonl $R/out_bench/*+gate.jsonl $R/out_gate_old/ 2>/dev/null  # superseded by +gate_k1
    rv() {  # rv <split-tag> <data> <cfg...>
      local t=$1 d=$2; shift 2
      for cfg in "$@"; do g=$(freest_gpu)
        run $g rev${t}_${cfg//[:+]/_} review.py --model $cfg --versions O --allow-unchecked --data $d --tag $t
      done
    }
    gt() {  # gt <split-tag> <data> <base> <gate args...>
      local t=$1 d=$2 b=$3; shift 3
      g=$(NEED_MIB=28000 freest_gpu)
      run $g gate${t}_${b//[:+]/_}_$(echo "$*" | tr -d ' -') gate.py --base $b --data $d --tag $t "$@"
    }
    VAR=("--k 1" "--k 2" "--k 3" "--k 1 --mode minus" "--k 2 --mode minus")
    ABL=("--k 1 --no-quote" "--k 1 --no-reported" "--k 1 --audit-samples 1" "--k 1 --checker qwen3-8b"
         "--k 2 --no-quote" "--k 2 --no-reported" "--k 2 --checker qwen3-8b")
    # phase 1: new reviews / audits (three lanes); lane C also builds the K1 and test check caches
    rv _dev bench_dev25 qwen3-32b-awq:think+audit &
    sleep 120
    { rv _dev bench_dev25 deepreviewer-7b:fast qwen3-32b-awq:s10 qwen3-32b-awq:ais
      rv _bench24 bench_iclr24 deepreviewer-7b:fast qwen3-32b-awq:s10 qwen3-32b-awq:ais; } &
    sleep 120
    { rv _bench bench_iclr25 qwen3-8b:think+audit
      gt "" versions deepreviewer-7b:fast --k 1
      gt _bench bench_iclr25 deepreviewer-7b:fast --k 1
      gt "" versions deepreviewer-7b:fast --k 1 --checker qwen3-8b
      gt _bench bench_iclr25 deepreviewer-7b:fast --k 1 --checker qwen3-8b
      rv _bench24 bench_iclr24 qwen3-32b-awq:think+audit; } &
    wait
    # phase 2: gate variants (checks are cached, so most of these take seconds)
    for b in deepreviewer-7b:fast qwen3-32b-awq:s10 qwen3-32b-awq:ais; do
      for v in "${VAR[@]}"; do gt _dev bench_dev25 $b $v; gt _bench24 bench_iclr24 $b $v; done
    done
    for b in deepreviewer-7b:fast qwen3-32b-awq:s10 qwen3-32b-awq:ais deepreviewer-7b:std qwen3-32b-awq:think+verify+s10; do
      for v in "${VAR[@]}"; do gt _bench bench_iclr25 $b $v; done
    done
    for b in deepreviewer-7b:fast qwen3-32b-awq:ais qwen3-32b-awq:think+verify+s10; do
      for v in "${VAR[@]}"; do gt "" versions $b $v; done
    done
    for b in deepreviewer-7b:fast qwen3-32b-awq:s10; do
      for v in "${ABL[@]}"; do gt _bench bench_iclr25 $b $v; done
      gt _bench bench_iclr25 $b --k 1 --audit qwen3-8b:think+audit; gt _bench bench_iclr25 $b --k 2 --audit qwen3-8b:think+audit
    done
    for b in deepreviewer-7b:fast qwen3-32b-awq:ais; do
      for v in "${ABL[@]}"; do gt "" versions $b $v; done
      gt "" versions $b --k 1 --audit qwen3-8b:think+audit; gt "" versions $b --k 2 --audit qwen3-8b:think+audit
    done
    # phase 3: evaluation
    $PY bench_eval.py --tag _dev --data bench_dev25 > $R/logs/eval_dev.log 2>&1
    $PY bench_eval.py --tag _bench --data bench_iclr25 > $R/logs/eval_bench.log 2>&1
    $PY bench_eval.py --tag _bench24 --data bench_iclr24 > $R/logs/eval_bench24.log 2>&1
    bash $0 judge
    echo "[$(date +%T)] paper stage done" >> $R/logs/k0.log ;;
  judge)
    g=$(NEED_MIB=28000 freest_gpu)
    K0_GPU_MEM=0.33 run $g judge judge.py --judge qwen3-32b-awq
    $PY analyze.py | tee $R/out/k0_result.txt ;;
esac
