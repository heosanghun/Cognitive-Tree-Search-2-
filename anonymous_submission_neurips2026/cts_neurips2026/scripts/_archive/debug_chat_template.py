"""Debug chat template to understand proper prompting."""
import multiprocessing as mp
import sys, os

def worker():
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root)
    os.chdir(root)
    
    import torch
    from cts.model.gemma_loader import load_gemma4_e4b
    
    mid = os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    model, tok = load_gemma4_e4b(
        model_id=mid, torch_dtype=torch.bfloat16,
        device_map="cuda:0", offload_vision_audio=True,
    )
    
    ct = getattr(tok, "chat_template", None)
    print(f"Has chat_template: {ct is not None}", flush=True)
    if ct:
        print(f"Template length: {len(ct)}", flush=True)
        print(f"Template preview:\n{ct[:500]}", flush=True)
    
    # Test with chat template
    messages = [{"role": "user", "content": "What is 15 + 27?"}]
    try:
        raw = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if isinstance(raw, torch.Tensor):
            ids = raw.to("cuda:0")
        else:
            ids = raw["input_ids"].to("cuda:0")
        
        decoded_input = tok.decode(ids[0], skip_special_tokens=False)
        print(f"\n=== Chat template input ===\n{decoded_input[:500]}", flush=True)
        
        pad = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0)
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, max_new_tokens=256, do_sample=False, pad_token_id=pad)
        out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"\n=== Chat template output ===\n{out[:500]}", flush=True)
    except Exception as e:
        print(f"Chat template error: {e}", flush=True)
    
    # Test with special tokens for thinking
    try:
        messages2 = [{"role": "user", "content": "Solve: What is the square root of 144?"}]
        raw2 = tok.apply_chat_template(messages2, add_generation_prompt=True, return_tensors="pt")
        if isinstance(raw2, torch.Tensor):
            ids2 = raw2.to("cuda:0")
        else:
            ids2 = raw2["input_ids"].to("cuda:0")
        with torch.inference_mode():
            gen2 = model.generate(input_ids=ids2, max_new_tokens=512, do_sample=False, pad_token_id=pad)
        out2 = tok.decode(gen2[0, ids2.shape[1]:], skip_special_tokens=True)
        print(f"\n=== Math with chat template ===\n{out2[:500]}", flush=True)
    except Exception as e:
        print(f"Math chat template error: {e}", flush=True)
    
    # Test GSM8K style with chat template
    try:
        messages3 = [{"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market for $2 per egg. How much does she make every day?"}]
        raw3 = tok.apply_chat_template(messages3, add_generation_prompt=True, return_tensors="pt")
        if isinstance(raw3, torch.Tensor):
            ids3 = raw3.to("cuda:0")
        else:
            ids3 = raw3["input_ids"].to("cuda:0")
        with torch.inference_mode():
            gen3 = model.generate(input_ids=ids3, max_new_tokens=512, do_sample=False, pad_token_id=pad)
        out3 = tok.decode(gen3[0, ids3.shape[1]:], skip_special_tokens=True)
        print(f"\n=== GSM8K with chat template ===\n{out3[:500]}", flush=True)
    except Exception as e:
        print(f"GSM8K chat template error: {e}", flush=True)
    
    # Test ARC style with chat template
    try:
        messages4 = [{"role": "user", "content": "An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\nA. Planetary density will decrease.\nB. Planetary years will become__(shorter/longer).\nC. Planetary days will become shorter.\nD. Planetary_(orbital speed) will increase.\nAnswer:"}]
        raw4 = tok.apply_chat_template(messages4, add_generation_prompt=True, return_tensors="pt")
        if isinstance(raw4, torch.Tensor):
            ids4 = raw4.to("cuda:0")
        else:
            ids4 = raw4["input_ids"].to("cuda:0")
        with torch.inference_mode():
            gen4 = model.generate(input_ids=ids4, max_new_tokens=64, do_sample=False, pad_token_id=pad)
        out4 = tok.decode(gen4[0, ids4.shape[1]:], skip_special_tokens=True)
        print(f"\n=== ARC with chat template ===\n{out4[:300]}", flush=True)
    except Exception as e:
        print(f"ARC chat template error: {e}", flush=True)
    
    print("\n=== DONE ===", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=worker)
    p.start()
    p.join()
    print(f"Exit code: {p.exitcode}")
