#!/usr/bin/env python3
"""Generates stratified problem IDs for math500 and gsm8k to ensure unbiased diagnosis.

MATH500: Stratifies by 'level' (1-5) and 'subject' (algebra, geometry, number theory, etc.).
GSM8K: Stratifies by hash bucketing to yield a uniform 20-sample subset.
"""
import json
from pathlib import Path
from typing import List, Dict

def stratify_math500(data_path: Path, n_samples: int = 20) -> List[str]:
    if not data_path.is_file():
        print(f"[warn] MATH500 file not found at {data_path}")
        return []
        
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            prob = json.loads(line)
            # Retrieve subject/level
            subject = prob.get("subject") or "unknown"
            level = prob.get("level") or 3
            unique_id = prob.get("unique_id") or prob.get("id") or str(i)
            rows.append({
                "id": str(unique_id),
                "subject": subject,
                "level": int(level)
            })
            
    # Bucket by (subject, level)
    buckets: Dict[str, List[str]] = {}
    for r in rows:
        key = f"{r['subject']}_L{r['level']}"
        buckets.setdefault(key, []).append(r["id"])
        
    # Stratified selection
    selected_ids = []
    bucket_keys = sorted(list(buckets.keys()))
    
    idx = 0
    while len(selected_ids) < n_samples and bucket_keys:
        key = bucket_keys[idx % len(bucket_keys)]
        if buckets[key]:
            # Pop the first element from this bucket
            selected_ids.append(buckets[key].pop(0))
        else:
            # Remove empty bucket
            bucket_keys.remove(key)
            if not bucket_keys:
                break
        idx += 1
        
    return selected_ids

def stratify_gsm8k(data_path: Path, n_samples: int = 20) -> List[str]:
    if not data_path.is_file():
        print(f"[warn] GSM8K file not found at {data_path}")
        return []
        
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            prob = json.loads(line)
            unique_id = prob.get("unique_id") or prob.get("id") or prob.get("idx") or str(i)
            rows.append(str(unique_id))
            
    # Simple stride to pick n uniform samples
    stride = max(1, len(rows) // n_samples)
    selected_ids = [rows[i] for i in range(0, len(rows), stride)[:n_samples]]
    return selected_ids

def main():
    root = Path(__file__).resolve().parent.parent
    
    # MATH500
    math_data = root / "data" / "math500" / "test.jsonl"
    math_ids = stratify_math500(math_data, 20)
    math_out = root / "data" / "math500_stratified_ids.txt"
    math_out.write_text("\n".join(math_ids), encoding="utf-8")
    print(f"Generated {len(math_ids)} stratified IDs for MATH500 -> {math_out}")
    
    # GSM8K
    gsm_data = root / "data" / "gsm8k" / "test.jsonl"
    if gsm_data.is_file():
        gsm_ids = stratify_gsm8k(gsm_data, 20)
        gsm_out = root / "data" / "gsm8k_stratified_ids.txt"
        gsm_out.write_text("\n".join(gsm_ids), encoding="utf-8")
        print(f"Generated {len(gsm_ids)} stratified IDs for GSM8K -> {gsm_out}")
    else:
        # GSM8K folder check
        gsm_dir = root / "data" / "gsm8k"
        print(f"[info] GSM8K folder contains: {[x.name for x in gsm_dir.iterdir() if x.is_file()]}")

if __name__ == "__main__":
    main()
