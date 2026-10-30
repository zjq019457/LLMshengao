"""Copy a Hugging Face model to the server one file at a time (the local disk cannot hold a whole model):
download a file, upload it with sftp, check the size on the server, delete the local copy. Resumable: files already
on the server with the right size are skipped. Uses the local Hugging Face login; no token is read or printed here.

  python stream_model.py <repo id> <server model dir name> <local tmp dir>
"""
import os
import shutil
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download

HOST = "ecs-bai-quad.fullerton.edu"
PATTERNS = (".json", ".safetensors", ".model", ".txt", ".jinja", ".py", ".md")


def ssh(cmd):
    return subprocess.run(["ssh", HOST, cmd], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120).stdout.strip()


def main(repo, name, tmp):
    dst = f"/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0/models/{name}"
    ssh(f"mkdir -p {dst}")
    info = HfApi().model_info(repo, files_metadata=True)
    files = [(s.rfilename, s.size) for s in info.siblings if s.rfilename.endswith(PATTERNS) and "/" not in s.rfilename]
    for fn, size in files:
        if ssh(f"stat -c %s {dst}/{fn} 2>/dev/null || echo 0") == str(size):
            print("skip", fn, flush=True)
            continue
        path = hf_hub_download(repo, fn, local_dir=tmp)
        for _ in range(5):
            r = subprocess.run(["sftp", "-q", "-b", "-", HOST], input=f'put "{path.replace(os.sep, "/")}" {dst}/{fn}\n', text=True)
            if r.returncode == 0 and ssh(f"stat -c %s {dst}/{fn}") == str(size):
                break
        else:
            sys.exit(f"upload failed: {fn}")
        os.remove(path)
        print("done", fn, size, flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    ssh(f"touch {dst}/UPLOAD_COMPLETE")
    print("ALL_DONE", repo)


if __name__ == "__main__":
    main(*sys.argv[1:4])
