"""Debug model outputs to improve answer extraction."""
import multiprocessing as mp
import sys, os

def worker():
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root)
    os.chdir(root)
    
    import json, torch
    from scripts.run_paper_reproduction import (
        load_model, generate_text, format_prompt,
        extract_final_answer, check_answer
    )
    
    model, tok = load_model()
    
    benchmarks = {
        "gsm8k": {"path": "data/gsm8k/test.jsonl", "q_key": "question", "a_key": "answer", "type": "gsm8k"},
        "arc": {"path": "data/arc_agi/test.jsonl", "q_key": "input", "a_key": "output", "type": "arc"},
        "math500": {"path": "data/math500/test.jsonl", "q_key": "problem", "a_key": "answer", "type": "math"},
    }
    
    for bname, info in benchmarks.items():
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {bname}")
        print(f"{'='*60}")
        
        problems = []
        with open(info["path"], "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        for i, prob in enumerate(problems[:3]):
            q = str(prob.get(info["q_key"], ""))
            gold = str(prob.get(info["a_key"], ""))
            prompt = format_prompt(q, info["type"])
            
            pred = generate_text(model, tok, prompt, max_tokens=512)
            
            pred_ans = extract_final_answer(pred, info["type"])
            gold_ans = extract_final_answer(gold, info["type"])
            correct = check_answer(pred, gold, info["type"])
            
            print(f"\n--- Problem {i+1} ---")
            print(f"Question: {q[:200]}...")
            print(f"Gold answer: {gold[:200]}")
            print(f"Gold extracted: {gold_ans}")
            print(f"Model output (first 300): {pred[:300]}")
            print(f"Pred extracted: {pred_ans}")
            print(f"Correct: {correct}")
        
        # Native think test for first problem
        prob = problems[0]
        q = str(prob.get(info["q_key"], ""))
        gold = str(prob.get(info["a_key"], ""))
        think_prompt = f"Think step by step carefully, then give the final answer.\n\n{format_prompt(q, info['type'])}"
        pred_think = generate_text(model, tok, think_prompt, max_tokens=1024)
        print(f"\n--- Native Think Sample ---")
        print(f"Output (first 500): {pred_think[:500]}")
        print(f"Extracted: {extract_final_answer(pred_think, info['type'])}")
        print(f"Correct: {check_answer(pred_think, gold, info['type'])}")
    
    print("\n=== DEBUG COMPLETE ===", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=worker)
    p.start()
    p.join()
    print(f"Exit code: {p.exitcode}")
