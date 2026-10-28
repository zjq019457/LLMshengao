#!/bin/bash
# Resumable upload of a local model folder to the server (sftp `reput` continues partial files).
# Usage: bash upload_model.sh <local dir> <server model dir name>; re-run until it prints ALL_DONE.
SRC=$1
DST=/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0/models/$2
H=ecs-bai-quad.fullerton.edu
ssh $H "mkdir -p $DST" < /dev/null || exit 1
for f in "$SRC"/*; do
  [ -f "$f" ] || continue
  n=$(basename "$f"); local_sz=$(stat -c %s "$f")
  remote_sz=$(ssh $H "stat -c %s $DST/$n 2>/dev/null || echo 0" < /dev/null)
  [ "$remote_sz" = "$local_sz" ] && continue
  echo "uploading $n ($remote_sz / $local_sz)"
  if [ "$remote_sz" = 0 ]; then cmd="put"; else cmd="reput"; fi
  echo "$cmd \"$f\" $DST/$n" | sftp -q -b - $H || exit 1
done
ssh $H "touch $DST/UPLOAD_COMPLETE" < /dev/null && echo ALL_DONE
