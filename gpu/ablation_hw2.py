"""HW2 ablation — all five configs measured back-to-back in ONE process/session,
so every speedup is a true ratio against the same-session baseline.

Run on the GPU box:
    python ablation_hw2.py

Mirrors the notebook harness exactly: same config, SEED=0, PROMPT_LEN=64,
MAX_NEW_TOKENS=128, same time_loop (CUDA events, 1 warmup / 3 iters).
"""
import time
import torch
from transformers import LlamaConfig, LlamaForCausalLM, StaticCache

SEED = 0
PROMPT_LEN = 64
MAX_NEW_TOKENS = 128


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _config():
    return LlamaConfig(
        vocab_size=32000, hidden_size=512, intermediate_size=1376,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=8,
        max_position_embeddings=4096,
    )


def build_model_and_input(dtype=torch.float32):
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(_config()).to(device=device(), dtype=dtype).eval()
    input_ids = torch.randint(0, 32000, (1, PROMPT_LEN), device=device())
    return model, input_ids


def time_loop(loop_fn, *args, n_warmup=1, n_iters=3):
    for _ in range(n_warmup):
        loop_fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            loop_fn(*args)
        end.record()
        torch.cuda.synchronize()
        return (start.elapsed_time(end) / 1e3) / n_iters
    t0 = time.perf_counter()
    for _ in range(n_iters):
        loop_fn(*args)
    return (time.perf_counter() - t0) / n_iters


# --- the five loops ---------------------------------------------------------
@torch.no_grad()
def baseline_loop(model, input_ids, n):
    """V0: fp32, NO KV cache, .item() host sync every step."""
    generated = input_ids
    out_tokens = []
    for _ in range(n):
        out = model(input_ids=generated, use_cache=False)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        _ = next_tok.item()
        generated = torch.cat([generated, next_tok], dim=1)
        out_tokens.append(next_tok)
    return torch.cat(out_tokens, dim=1)


@torch.no_grad()
def kv_loop(model, input_ids, n):
    """+ KV cache, .item() removed (HF dynamic cache)."""
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    toks = [next_tok]
    for _ in range(n - 1):
        out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        toks.append(next_tok)
    return torch.cat(toks, dim=1)


@torch.no_grad()
def static_compiled_loop(model, input_ids, n):
    """+ StaticCache + compile(reduce-overhead): one CUDA-graph replay per step."""
    total_len = PROMPT_LEN + n
    cache = StaticCache(config=model.config, max_batch_size=1,
                        max_cache_len=total_len, device=model.device,
                        dtype=model.dtype)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    tok = out.logits[:, -1:, :].argmax(dim=-1).clone()
    toks = [tok]
    for _ in range(n - 1):
        out = model(input_ids=tok, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1).clone()
        toks.append(tok)
    return torch.cat(toks, dim=1)


def main():
    n = MAX_NEW_TOKENS
    rows = []

    # 1. baseline (fp32, no KV)
    m, ids = build_model_and_input(torch.float32)
    rows.append(("Baseline: fp32, no KV cache, .item() each step",
                 time_loop(baseline_loop, m, ids, n)))

    # 2. + KV cache (fp32)
    m, ids = build_model_and_input(torch.float32)
    rows.append(("+ KV cache (fp32, .item() removed)",
                 time_loop(kv_loop, m, ids, n)))

    # 3. + KV + bf16 (no compile)
    m, ids = build_model_and_input(torch.bfloat16)
    rows.append(("+ KV + bf16 (no compile)",
                 time_loop(kv_loop, m, ids, n)))

    # 4. + KV + bf16 + torch.compile(dynamic=True)
    m, ids = build_model_and_input(torch.bfloat16)
    cm = torch.compile(m, dynamic=True)
    for _ in range(3):                      # let dynamic compile settle
        kv_loop(cm, ids, n)
    torch.cuda.synchronize()
    rows.append(("+ KV + bf16 + torch.compile(dynamic=True)",
                 time_loop(kv_loop, cm, ids, n)))

    # 5. + StaticCache + bf16 + compile(reduce-overhead)
    m, ids = build_model_and_input(torch.bfloat16)
    cm = torch.compile(m, mode="reduce-overhead")
    for _ in range(3):                      # finish CUDA-graph capture
        static_compiled_loop(cm, ids, n)
    torch.cuda.synchronize()
    rows.append(("+ StaticCache + bf16 + compile(reduce-overhead)",
                 time_loop(static_compiled_loop, cm, ids, n)))

    base = rows[0][1]
    print(f"\ndevice={torch.cuda.get_device_name(0)}  "
          f"prompt={PROMPT_LEN}  new_tokens={n}  (one session)\n")
    print(f"{'Configuration':52s} {'ms':>8s} {'speedup':>9s}")
    print("-" * 72)
    for name, t in rows:
        print(f"{name:52s} {t*1e3:8.1f} {base/t:8.2f}x")
    print("\nMarkdown rows:")
    for name, t in rows:
        print(f"| {name} | {t*1e3:.1f} | {base/t:.2f}× |")


if __name__ == "__main__":
    main()
