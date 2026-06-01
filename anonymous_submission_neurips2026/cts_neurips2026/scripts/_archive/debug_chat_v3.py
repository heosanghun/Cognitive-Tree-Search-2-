"""Deep debug: check raw tokens, special tokens, generation behavior."""
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
    
    # Check special tokens
    print(f"EOS token id: {tok.eos_token_id}", flush=True)
    print(f"EOS token: {tok.eos_token}", flush=True)
    print(f"BOS token id: {tok.bos_token_id}", flush=True)
    print(f"PAD token id: {tok.pad_token_id}", flush=True)
    
    # Check if turn tokens exist
    for token_str in ["<|turn>", "<turn|>", "<|turn>model", "<|turn>user"]:
        ids = tok.encode(token_str, add_special_tokens=False)
        print(f"Token '{token_str}' -> ids: {ids}", flush=True)
    
    # Test with raw decode (no skip_special_tokens)
    prompt = "What is 15 + 27?"
    messages = [{"role": "user", "content": prompt}]
    ids_raw = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if isinstance(ids_raw, torch.Tensor):
        ids = ids_raw.to("cuda:0")
    elif hasattr(ids_raw, "input_ids"):
        ids = ids_raw.input_ids.to("cuda:0")
    else:
        ids = ids_raw["input_ids"].to("cuda:0") if isinstance(ids_raw, dict) else torch.tensor([ids_raw], device="cuda:0")
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    in_len = ids.shape[1]
    
    print(f"\nInput decoded (raw): {tok.decode(ids[0], skip_special_tokens=False)}", flush=True)
    print(f"Input length: {in_len}", flush=True)
    
    with torch.inference_mode():
        gen = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False, pad_token_id=pad)
    
    raw_out = tok.decode(gen[0, in_len:], skip_special_tokens=False)
    clean_out = tok.decode(gen[0, in_len:], skip_special_tokens=True)
    print(f"Raw output (with special): [{raw_out[:300]}]", flush=True)
    print(f"Clean output: [{clean_out[:300]}]", flush=True)
    print(f"Generated tokens: {gen[0, in_len:].tolist()[:20]}", flush=True)
    
    # Test WITHOUT chat template - just raw prompt
    prompt2 = "Q: What is 15 + 27?\nA:"
    enc2 = tok(prompt2, return_tensors="pt", truncation=True, max_length=4096)
    ids2 = enc2["input_ids"].to("cuda:0")
    in_len2 = ids2.shape[1]
    with torch.inference_mode():
        gen2 = model.generate(input_ids=ids2, max_new_tokens=128, do_sample=False, pad_token_id=pad)
    raw2 = tok.decode(gen2[0, in_len2:], skip_special_tokens=False)
    clean2 = tok.decode(gen2[0, in_len2:], skip_special_tokens=True)
    print(f"\nQ&A format raw: [{raw2[:300]}]", flush=True)
    print(f"Q&A format clean: [{clean2[:300]}]", flush=True)
    
    # Test few-shot
    prompt3 = """Q: What is 5 + 3?
A: 8

Q: What is 10 * 4?
A: 40

Q: What is 15 + 27?
A:"""
    enc3 = tok(prompt3, return_tensors="pt", truncation=True, max_length=4096)
    ids3 = enc3["input_ids"].to("cuda:0")
    in_len3 = ids3.shape[1]
    with torch.inference_mode():
        gen3 = model.generate(input_ids=ids3, max_new_tokens=64, do_sample=False, pad_token_id=pad)
    out3 = tok.decode(gen3[0, in_len3:], skip_special_tokens=True)
    print(f"\nFew-shot: [{out3[:200]}]", flush=True)
    
    # Test GSM8K with proper format
    prompt4 = """Problem: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Solution: He writes each friend 3*2=6 pages a week. So he writes 6*2=12 pages a week. That means he writes 12*52=624 pages a year.
#### 624

Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest for $2 per egg. How much does she make every day?
Solution:"""
    enc4 = tok(prompt4, return_tensors="pt", truncation=True, max_length=4096)
    ids4 = enc4["input_ids"].to("cuda:0")
    in_len4 = ids4.shape[1]
    with torch.inference_mode():
        gen4 = model.generate(input_ids=ids4, max_new_tokens=256, do_sample=False, pad_token_id=pad)
    out4 = tok.decode(gen4[0, in_len4:], skip_special_tokens=True)
    print(f"\nGSM8K few-shot: [{out4[:400]}]", flush=True)
    
    print("\n=== DONE ===", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=worker)
    p.start()
    p.join()
    print(f"Exit code: {p.exitcode}")
