#!/usr/bin/env python3
"""
Multiprocessing wrapper for run_paper_reproduction.py.
Bypasses Windows DLL deadlock issue with torch import in PowerShell.

Usage:
  python -u scripts/run_reproduction_mp.py --phase eval --eval-limit 50
  python -u scripts/run_reproduction_mp.py --phase all --ppo-steps 10000
"""
import multiprocessing as mp
import sys
import os

def worker(args_list):
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUNBUFFERED"] = "1"
    
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "scripts"))
    os.chdir(root)
    
    # Set unbuffered via env (fdopen causes issues in mp.Process on Windows)
    
    sys.argv = ["run_paper_reproduction.py"] + args_list
    
    import importlib
    mod = importlib.import_module("run_paper_reproduction")
    mod.main()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = sys.argv[1:]
    print(f"Launching reproduction pipeline in subprocess: {args}", flush=True)
    p = mp.Process(target=worker, args=(args,))
    p.start()
    p.join()
    print(f"\nProcess exited with code: {p.exitcode}", flush=True)
    sys.exit(p.exitcode or 0)
