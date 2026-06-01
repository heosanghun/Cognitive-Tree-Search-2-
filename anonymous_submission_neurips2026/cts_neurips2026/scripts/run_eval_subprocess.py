"""Launch run_paper_reproduction.py as a completely independent subprocess."""
import subprocess
import sys
import os

if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    args = sys.argv[1:]
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_MODULE_LOADING"] = "LAZY"
    
    cmd = [sys.executable, "-u", os.path.join(root, "scripts", "run_paper_reproduction.py")] + args
    print(f"Running: {' '.join(cmd)}", flush=True)
    
    proc = subprocess.Popen(
        cmd, cwd=root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    
    for line in proc.stdout:
        try:
            print(line.decode("utf-8", errors="replace"), end="", flush=True)
        except Exception:
            pass
    
    proc.wait()
    print(f"\nExit code: {proc.returncode}", flush=True)
    sys.exit(proc.returncode)
