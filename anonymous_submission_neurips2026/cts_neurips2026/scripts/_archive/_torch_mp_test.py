"""Test torch import via multiprocessing to bypass DLL deadlock."""
import multiprocessing as mp
import sys
import os

def worker():
    print("worker: importing torch...", flush=True)
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    import torch
    print(f"worker: torch OK, CUDA={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"worker: GPU={torch.cuda.get_device_name(0)}", flush=True)
    print("worker: SUCCESS", flush=True)

if __name__ == "__main__":
    print("main: starting worker process", flush=True)
    p = mp.Process(target=worker)
    p.start()
    p.join(timeout=60)
    if p.is_alive():
        print("main: worker timed out, killing", flush=True)
        p.kill()
        p.join()
        print("main: FAILED - torch import hung", flush=True)
        sys.exit(1)
    else:
        print(f"main: worker exit code={p.exitcode}", flush=True)
        if p.exitcode == 0:
            print("main: SUCCESS", flush=True)
        else:
            print("main: FAILED", flush=True)
            sys.exit(1)
