#!/bin/bash
# Resumable upload of DeepReviewer-7B (D:\123) to the server: sftp `reput` continues partial files.
# Re-run until it prints ALL_DONE.
SRC=/d/123
DST=/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0/models/DeepReviewer-7B
H=ecs-bai-quad.fullerton.edu
ssh $H "mkdir -p $DST" || exit 1
for f in "$SRC"/*; do
  n=$(basename "$f"); local_sz=$(stat -c %s "$f")
  remote_sz=$(ssh $H "stat -c %s $DST/$n 2>/dev/null || echo 0")
  [ "$remote_sz" = "$local_sz" ] && continue
  echo "uploading $n ($remote_sz / $local_sz)"
  if [ "$remote_sz" = 0 ]; then cmd="put"; else cmd="reput"; fi
  echo "$cmd \"$f\" $DST/$n" | sftp -q -b - $H || exit 1
done
ssh $H "touch $DST/UPLOAD_COMPLETE" && echo ALL_DONE
