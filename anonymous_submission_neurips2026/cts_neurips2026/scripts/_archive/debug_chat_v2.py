"""Test generation with manually applied chat template."""
import multiprocessing as mp
import sys, os

def worker():
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root)
    os.chdir(root)
    
    import torch
    from cts.model.gemma_loader import load_gemma4_e4b
    
    GEMMA4_CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|turn>system\n{{ message['content'] }}<turn|>\n"
        "{% elif message['role'] == 'user' %}<|turn>user\n{{ message['content'] }}<turn|>\n"
        "{% elif message['role'] == 'model' or message['role'] == 'assistant' %}<|turn>model\n{{ message['content'] }}<turn|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
    )
    
    mid = os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
    model, tok = load_gemma4_e4b(
        model_id=mid, torch_dtype=torch.bfloat16,
        device_map="cuda:0", offload_vision_audio=True,
    )
    tok.chat_template = GEMMA4_CHAT_TEMPLATE
    pad = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0)
    
    def gen_chat(prompt, max_tokens=512):
        messages = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if isinstance(ids, torch.Tensor):
            ids = ids.to("cuda:0")
        else:
            ids = ids["input_ids"].to("cuda:0")
        in_len = ids.shape[1]
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, max_new_tokens=max_tokens, do_sample=False, pad_token_id=pad)
        return tok.decode(gen[0, in_len:], skip_special_tokens=True)
    
    # Test 1: Simple math
    out = gen_chat("What is 15 + 27? Answer with just the number.")
    print(f"[Test 1] 15+27: {out[:200]}", flush=True)
    
    # Test 2: GSM8K style
    out2 = gen_chat("Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market for $2 per egg. How much does she make every day? Show your work step by step, then give the final answer after ####.")
    print(f"\n[Test 2] GSM8K: {out2[:500]}", flush=True)
    
    # Test 3: MATH boxed
    out3 = gen_chat("Solve the following math problem step by step. Put your final answer in \\boxed{}.\n\nConvert the point (0,3) in rectangular coordinates to polar coordinates. Enter your answer in the form (r,theta), where r > 0 and 0 <= theta < 2*pi.")
    print(f"\n[Test 3] MATH boxed: {out3[:500]}", flush=True)
    
    # Test 4: ARC
    out4 = gen_chat("An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?\nA. Planetary density will decrease.\nB. Planetary years will become longer.\nC. Planetary days will become shorter.\nD. Planetary orbital speed will increase.\n\nAnswer with just the letter.")
    print(f"\n[Test 4] ARC: {out4[:200]}", flush=True)
    
    # Test 5: HumanEval
    out5 = gen_chat("Write a Python function that takes a list of numbers and returns the sum of all even numbers in the list. Only output the function body, no explanation.")
    print(f"\n[Test 5] Code: {out5[:300]}", flush=True)
    
    print("\n=== CHAT TEMPLATE TEST DONE ===", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=worker)
    p.start()
    p.join()
    print(f"Exit code: {p.exitcode}")
