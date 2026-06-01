#!/usr/bin/env python3
"""Diagnose model generation quality to understand low benchmark scores."""
import os, sys, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cts.model.gemma_loader import load_gemma4_e4b

mid = os.environ.get("CTS_GEMMA_MODEL_DIR", "google/gemma-4-E4B")
print(f"Model ID: {mid}")

model, tok = load_gemma4_e4b(
    model_id=mid, torch_dtype=torch.bfloat16,
    device_map="cuda:0", offload_vision_audio=True,
)
print(f"Model class: {type(model).__name__}")

ct = getattr(tok, "chat_template", None)
has_ct = ct is not None and isinstance(ct, str) and len(ct) > 0
print(f"Has chat_template: {has_ct}")
if has_ct:
    print(f"chat_template length: {len(ct)}")

# Test 1: Raw prompt
prompt = "What is 15 + 27? Give only the number."
enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
ids = enc["input_ids"].to("cuda:0")
pad = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0)
with torch.inference_mode():
    gen = model.generate(input_ids=ids, max_new_tokens=128, do_sample=False, pad_token_id=pad)
out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
print(f"\n[Test 1] Raw prompt output:\n{out[:500]}")

# Test 2: Chat template
try:
    messages = [{"role": "user", "content": "What is 15 + 27? Give only the number."}]
    raw = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if isinstance(raw, torch.Tensor):
        ids2 = raw.to("cuda:0")
    else:
        ids2 = raw["input_ids"].to("cuda:0")
    with torch.inference_mode():
        gen2 = model.generate(input_ids=ids2, max_new_tokens=128, do_sample=False, pad_token_id=pad)
    out2 = tok.decode(gen2[0, ids2.shape[1]:], skip_special_tokens=True)
    print(f"\n[Test 2] Chat template output:\n{out2[:500]}")
except Exception as e:
    print(f"\n[Test 2] Chat template error: {e}")

# Test 3: MATH-style with boxed
prompt3 = "Solve the following math problem step by step.\n\nConvert the point (0,3) in rectangular coordinates to polar coordinates.\n\nPut your final answer in \\boxed{}."
enc3 = tok(prompt3, return_tensors="pt", truncation=True, max_length=4096)
ids3 = enc3["input_ids"].to("cuda:0")
with torch.inference_mode():
    gen3 = model.generate(input_ids=ids3, max_new_tokens=512, do_sample=False, pad_token_id=pad)
out3 = tok.decode(gen3[0, ids3.shape[1]:], skip_special_tokens=True)
print(f"\n[Test 3] MATH boxed output:\n{out3[:800]}")

# Test 4: GSM8K style
prompt4 = "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market for $2 per egg. How much does she make every day? Think step by step."
enc4 = tok(prompt4, return_tensors="pt", truncation=True, max_length=4096)
ids4 = enc4["input_ids"].to("cuda:0")
with torch.inference_mode():
    gen4 = model.generate(input_ids=ids4, max_new_tokens=512, do_sample=False, pad_token_id=pad)
out4 = tok.decode(gen4[0, ids4.shape[1]:], skip_special_tokens=True)
print(f"\n[Test 4] GSM8K output:\n{out4[:800]}")

# Test 5: ARC style
prompt5 = "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\nA. dry palms\nB. wet palms\nC. palms covered with oil\nD. palms covered with lotion\nAnswer:"
enc5 = tok(prompt5, return_tensors="pt", truncation=True, max_length=4096)
ids5 = enc5["input_ids"].to("cuda:0")
with torch.inference_mode():
    gen5 = model.generate(input_ids=ids5, max_new_tokens=64, do_sample=False, pad_token_id=pad)
out5 = tok.decode(gen5[0, ids5.shape[1]:], skip_special_tokens=True)
print(f"\n[Test 5] ARC output:\n{out5[:300]}")

print("\n=== DIAGNOSIS COMPLETE ===")
