#!/usr/bin/env python3
"""
End-to-end: Table 1 CSVs, Table 2 logs + Iso-FLOP JSON, optional Stage1/2 GPU training → artifacts/.

Tiers (default: quick — suitable for CI/local smoke on a 4090-class GPU):
  quick     — small depth sweep, MATH limit 24, Stage1 5 steps, Stage2 2 steps
  standard  — runbook depths, MATH 200, Stage1 80, Stage2 24
  full      — runbook depths, MATH 500; Stage1/2 steps & broyden from `--config` (default.yaml);
              use `--config paper_parity` for appendix-length training + parallel DEQ flags

Environment:
  HF_TOKEN        — if gated models are used
  CTS_GEMMA_MODEL_DIR — local Gemma folder (recommended)
  HF_HUB_CACHE    — defaults to repo `.hf_cache`

Usage:
  python scripts/run_paper_artifacts_pipeline.py
  python scripts/run_paper_artifacts_pipeline.py --tier standard --skip-training
  python scripts/run_paper_artifacts_pipeline.py --tier full --config paper_parity
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_train_cfg(root: Path, name: str) -> dict:
    if yaml is None:
        return {}
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from cts.utils.config import load_config

        return load_config(name)
    except Exception as e:
        print("WARN: load_config failed:", name, e)
        return {}


def _ensure_env() -> None:
    root = _root()
    os.environ.setdefault("HF_HUB_CACHE", str(root / ".hf_cache"))
    (root / ".hf_cache").mkdir(parents=True, exist_ok=True)
    gemma_it = root / "gemma-4-E4B-it"
    gemma_b = root / "gemma-4-E4B"
    if not os.environ.get("CTS_GEMMA_MODEL_DIR"):
        if gemma_it.is_dir() and (gemma_it / "config.json").is_file():
            os.environ["CTS_GEMMA_MODEL_DIR"] = str(gemma_it)
        elif gemma_b.is_dir() and (gemma_b / "config.json").is_file():
            os.environ["CTS_GEMMA_MODEL_DIR"] = str(gemma_b)


def _write_math500_metrics_json(
    log_path: Path,
    out_path: Path,
    *,
    limit: int,
    structured_json: Path | None = None,
) -> None:
    """Prefer `--out-json` from run_math500; else parse final dict line from log."""
    if structured_json is not None and structured_json.is_file():
        data = json.loads(structured_json.read_text(encoding="utf-8"))
        res = data.get("result", {})
        summary = {k: v for k, v in res.items() if k != "items"}
        payload = {
            "script": "scripts/run_math500.py",
            "limit": limit,
            "source": "structured_json",
            "structured": str(structured_json),
            "result": summary,
            "n_items": len(res.get("items", [])),
            "log": str(log_path),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    try:
        txt = log_path.read_text(encoding="utf-8")
    except OSError:
        return
    result = None
    for line in reversed(txt.splitlines()):
        s = line.strip()
        if s.startswith("{") and "pass_at_1" in s:
            try:
                result = ast.literal_eval(s)
            except (SyntaxError, ValueError):
                continue
            break
    payload = {
        "script": "scripts/run_math500.py",
        "limit": limit,
        "flags": ["--gemma", "--think-prompt", "--chat-template"],
        "result": result,
        "log": str(log_path),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    log_file: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, **(env or {})}
    t0 = time.perf_counter()
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as lf:
            lf.write(f"# cmd: {' '.join(cmd)}\n# cwd: {cwd}\n\n")
            p = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=full_env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
    else:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=full_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    p.duration_s = time.perf_counter() - t0  # type: ignore[attr-defined]
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("quick", "standard", "full"), default="quick")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-math-gemma", action="store_true", help="Skip slow Gemma MATH eval")
    ap.add_argument("--skip-arc", action="store_true", default=True, help="Skip ARC (needs JSONL path)")
    ap.add_argument("--arc-data", type=str, default=None, help="Optional ARC JSONL for table2")
    ap.add_argument(
        "--config",
        type=str,
        default="default",
        help="YAML merged with default.yaml; full tier uses stage1_max_steps, stage2_total_ppo_steps, broyden_max_iter",
    )
    args = ap.parse_args()

    root = _root()
    py = sys.executable
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    _ensure_env()

    cfg_train = _load_train_cfg(root, args.config)
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from cts.utils.repro_snapshot import write_repro_snapshot

        write_repro_snapshot(art / "REPRO_ENV.json", root=root)
    except Exception as e:
        print("WARN: REPRO_ENV.json not written:", e)

    tier = args.tier
    s2_ppo_epochs = 1
    if tier == "quick":
        depths = [1, 5, 10, 15, 20]
        depths_kv = [1, 5, 10, 15]
        math_limit = 24
        s1_steps = 5
        s2_steps = 2
        s2_batch = 1
        broyden = 6
    elif tier == "standard":
        depths = [1, 5, 10, 15, 20]
        depths_kv = [1, 5, 10, 15]
        math_limit = 200
        s1_steps = 80
        s2_steps = 24
        s2_batch = 2
        broyden = 10
    else:
        depths = [1, 5, 10, 15, 20, 25, 30]
        depths_kv = [1, 5, 10, 15, 20]
        math_limit = 500
        if cfg_train:
            s1_steps = int(cfg_train.get("stage1_max_steps", 500))
            s2_steps = int(cfg_train.get("stage2_total_ppo_steps", 100))
            s2_batch = int(cfg_train.get("pipeline_stage2_collect_batch", 2))
            broyden = int(cfg_train.get("broyden_max_iter", 30))
            s2_ppo_epochs = int(cfg_train.get("pipeline_stage2_ppo_epochs", 2))
        else:
            s1_steps = 500
            s2_steps = 100
            s2_batch = 2
            broyden = 12

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
    except Exception:
        cuda_ok = False
        gpu_name = None

    manifest: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "config": args.config,
        "cuda_available": cuda_ok,
        "gpu_name": gpu_name,
        "cts_gemma_model_dir": os.environ.get("CTS_GEMMA_MODEL_DIR"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "steps": [],
    }

    def record(name: str, p: subprocess.CompletedProcess[str], extra: dict | None = None) -> None:
        entry = {
            "name": name,
            "returncode": p.returncode,
            "duration_s": round(getattr(p, "duration_s", 0.0), 3),
        }
        if extra:
            entry.update(extra)
        manifest["steps"].append(entry)

    # 1) Data
    if not args.skip_download:
        dl_cmd = [
            py,
            str(root / "scripts" / "download_experiment_data.py"),
            "--openmath-rows",
            "8000" if tier == "quick" else "100000",
            "--math-train-rows",
            "5000",
        ]
        p = _run(dl_cmd, cwd=root, log_file=art / "log_download_experiment_data.txt")
        record("download_experiment_data", p)
        if p.returncode != 0:
            print("WARN: download failed — continuing if local JSONL exists. Log:", art / "log_download_experiment_data.txt")

    # 2) Table 1 — analytic + timing
    t1_out = art / "table1_cts_kv.csv"
    p = _run(
        [
            py,
            "-m",
            "cts.eval.profile_vram_latency",
            "--depths",
            *[str(d) for d in depths],
            "--out",
            str(t1_out),
            "--cuda",
        ],
        cwd=root,
        log_file=art / "log_table1_profile_vram_latency.txt",
    )
    record("profile_vram_latency", p, {"out": str(t1_out)})

    # 3) Table 1 — measured KV
    t1_kv = art / "table1_kv_measured.csv"
    p = _run(
        [
            py,
            str(root / "scripts" / "profile_kv_measured.py"),
            "--depths",
            *[str(d) for d in depths_kv],
            "--out",
            str(t1_kv),
        ],
        cwd=root,
        log_file=art / "log_table1_kv_measured.txt",
    )
    record("profile_kv_measured", p, {"out": str(t1_kv)})

    # 4) Table 2 — Iso-FLOP (mock backbone JSON)
    iso_path = art / "table2_isoflop_mock.json"
    t0 = time.perf_counter()
    pr = subprocess.run(
        [py, "-m", "cts.eval.report_isoflop", "--json"],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if pr.returncode == 0 and pr.stdout.strip():
        iso_path.write_text(pr.stdout, encoding="utf-8")
    else:
        iso_path.write_text(
            json.dumps({"returncode": pr.returncode, "stderr": pr.stderr, "stdout": pr.stdout}, indent=2),
            encoding="utf-8",
        )
    pr.duration_s = time.perf_counter() - t0  # type: ignore[attr-defined]
    record("report_isoflop", pr, {"out": str(iso_path)})

    # 5) Table 2 — MATH (Gemma)
    math_jsonl = root / "data" / "math500" / "test.jsonl"
    if not args.skip_math_gemma and math_jsonl.is_file():
        math_log = art / "table2_math500_gemma_log.txt"
        math_struct = art / "table2_math500_run.json"
        m_cmd = [
            py,
            str(root / "scripts" / "run_math500.py"),
            "--data",
            str(math_jsonl),
            "--gemma",
            "--limit",
            str(math_limit),
            "--think-prompt",
            "--chat-template",
            "--out-json",
            str(math_struct),
        ]
        p = _run(m_cmd, cwd=root, log_file=math_log)
        mjson = art / "table2_math500_metrics.json"
        _write_math500_metrics_json(math_log, mjson, limit=math_limit, structured_json=math_struct)
        record(
            "run_math500_gemma",
            p,
            {
                "limit": math_limit,
                "log": str(math_log),
                "structured_json": str(math_struct),
                "metrics": str(mjson),
            },
        )
    else:
        manifest["steps"].append({"name": "run_math500_gemma", "skipped": True, "reason": "no data or --skip-math-gemma"})

    # 6) ARC (optional)
    if args.arc_data and Path(args.arc_data).is_file():
        arc_log = art / "table2_arc_gemma_log.txt"
        arc_json = art / "table2_arc_run.json"
        p = _run(
            [
                py,
                str(root / "scripts" / "run_arc_agi_text.py"),
                "--data",
                args.arc_data,
                "--gemma",
                "--limit",
                str(min(200, math_limit)),
                "--think-prompt",
                "--chat-template",
                "--out-json",
                str(arc_json),
            ],
            cwd=root,
            log_file=arc_log,
        )
        record("run_arc_gemma", p, {"log": str(arc_log), "structured_json": str(arc_json)})

    # 7) Stage 1 / 2 training (real GPU steps)
    if not args.skip_training:
        if yaml is None:
            manifest["steps"].append({"name": "training", "skipped": True, "reason": "pyyaml missing"})
        else:
            dp = yaml.safe_load((root / "configs" / "data_paths.yaml").read_text(encoding="utf-8"))
            om_path = root / dp.get("openmath_train_jsonl", "data/openmath_instruct/train_100000.jsonl")
            if not om_path.is_file():
                om_alt = root / "data" / "openmath_instruct" / "train_10000.jsonl"
                om_path = om_alt if om_alt.is_file() else om_path

            s2_path = root / dp.get("stage2_math_prompts_jsonl", "data/stage2/math_train_prompts_5000.jsonl")

            if om_path.is_file():
                s1_cmd = [
                    py,
                    str(root / "scripts" / "run_stage1_openmath.py"),
                    "--config",
                    args.config,
                    "--data",
                    str(om_path),
                    "--log-every",
                    "1",
                ]
                if s1_steps > 0:
                    s1_cmd.extend(["--max-steps", str(s1_steps)])
                train_env = {}
                if cfg_train.get("cts_deq_map_mode"):
                    train_env["CTS_DEQ_MAP_MODE"] = str(cfg_train["cts_deq_map_mode"])
                p = _run(s1_cmd, cwd=root, log_file=art / "log_stage1_openmath.txt", env=train_env)
                record("run_stage1_openmath", p)
            else:
                manifest["steps"].append({"name": "run_stage1_openmath", "skipped": True, "reason": f"missing {om_path}"})

            ckpt = art / "stage1_last.pt"
            if s2_path.is_file() and ckpt.is_file():
                s2_cmd = [
                    py,
                    str(root / "scripts" / "run_stage2_math_ppo.py"),
                    "--config",
                    args.config,
                    "--data",
                    str(s2_path),
                    "--stage1-ckpt",
                    str(ckpt),
                    "--collect-batch",
                    str(s2_batch),
                    "--broyden-max-iter",
                    str(broyden),
                    "--log-every",
                    "1",
                    "--ppo-epochs",
                    str(s2_ppo_epochs),
                ]
                if s2_steps > 0:
                    s2_cmd.extend(["--steps", str(s2_steps)])
                if cfg_train.get("stage2_parallel_map"):
                    s2_cmd.append("--parallel-map")
                smoke_env = {}
                if tier in ("quick", "standard"):
                    smoke_env["CTS_STAGE2_SMOKE"] = "1"
                train_env = {}
                if cfg_train.get("cts_deq_map_mode"):
                    train_env["CTS_DEQ_MAP_MODE"] = str(cfg_train["cts_deq_map_mode"])
                p = _run(
                    s2_cmd,
                    cwd=root,
                    log_file=art / "log_stage2_math_ppo.txt",
                    env={**smoke_env, **train_env},
                )
                record("run_stage2_math_ppo", p)
            else:
                manifest["steps"].append(
                    {
                        "name": "run_stage2_math_ppo",
                        "skipped": True,
                        "reason": f"ckpt={ckpt.is_file()} data={s2_path.is_file()}",
                    }
                )

    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    man_path = art / "RUN_MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Manifest: {man_path}")
    print("Artifacts under:", art)


if __name__ == "__main__":
    main()
