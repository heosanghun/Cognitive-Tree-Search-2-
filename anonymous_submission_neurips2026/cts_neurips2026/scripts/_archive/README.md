# Archived Scripts

This folder contains scratch/debug scripts that were used during development
but are **not** part of the documented reproduction pipeline. They are kept
in the repo for historical context and to make development decisions
auditable, but reviewers can safely ignore them.

| Script | Purpose during development |
|:---|:---|
| `_test_torch.py` / `_torch_mp_test.py` | Quick sanity checks for torch and torch.multiprocessing on a fresh environment. |
| `debug_chat_template.py`, `debug_chat_v2.py`, `debug_chat_v3.py`, `show_chat_template.py` | Iteratively diagnosing how Gemma 4 E4B's `apply_chat_template` interacts with `enable_thinking`. The conclusions are baked into `cts/eval/think_prompt.py` and `_build_prompt()` in `scripts/run_cts_eval_full.py`. |
| `debug_outputs.py` / `diagnose_model.py` | Inspecting raw `model.generate` outputs while bringing up the Gemma backbone. The eval pipeline now uses `cts/eval/gemma_predict.GemmaTextPredictor` directly. |

If you are looking for the canonical evaluation entry point, see
[`../run_cts_eval_full.py`](../run_cts_eval_full.py).
