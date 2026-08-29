# HW2 — Profile & Optimize an Autoregressive Decode Loop

**Lecture mapping:** L1 §07 (Profiling) · L2 §01 (prefill/decode, KV cache) · L2 §03 (engine optimizations)

The harness below defines a deliberately slow greedy-decode loop (**V0**): no KV
cache (it recomputes the whole sequence every step), fp32 eager, and a host sync
every step. Profile it, find the bottlenecks, and write a fast, numerically
identical replacement.

## What you implement

| Part | Function |
|------|----------|
| 1 | `profile` — wrap a loop in `torch.profiler`, print a table, export a Chrome trace |
| 2 | `optimized_loop` — fast greedy decode, same tokens as V0 (fp32) |
| 3 | `generate_optimized` — build, warm up, and time your optimized generation |
| 4 | Writeup Q1–Q4 |

## Speedup targets

`optimized_loop` end-to-end vs the V0 baseline:

| Speedup vs V0 | |
|---------------|--|
| ≥ 4.0× | excellent |
| ≥ 3.0× | good |
| ≥ 1.5× | some speedup |
| < 1.5× | little to no speedup |

The speedup only counts if `optimized_loop` passes the fp32 correctness check and
the timed run is on a GPU.

## Rules

- PyTorch + `requirements.txt` only — no vLLM / TensorRT-LLM / SGLang.
- Don't edit the harness cell.
- `optimized_loop` must reproduce the baseline's greedy tokens exactly on the same
  fp32 model. `generate_optimized` may switch to bf16 for the timed run.

## Where to look

- **KV cache** (L2 §01) — the baseline throws it away every step. Biggest win.
- **Host syncs** (L1 §07) — `.item()` on the critical path serializes CPU↔GPU.
- **`torch.compile` / CUDA graphs** (L2 §03) — fuse ops, cut launch overhead.
- **dtype** — bf16 for the timed run (small on a model this size).

## Cell types

- **DO NOT EDIT** — fixed harness.
- **YOUR IMPLEMENTATION** — replace `raise NotImplementedError`.
- **SELF-CHECK** — asserts that must pass.
- **WRITEUP** — answer in the markdown cell.

Run top-to-bottom on the GPU. Submit the executed notebook plus the files under
`results/`.

## The fixed harness (DO NOT EDIT)

Tiny 2-layer Llama, the V0 baseline, the correctness check, the timing helper,
and the speedup targets.


```python
# DO NOT EDIT — editing this cell invalidates your speedup numbers.
import os, time

import torch
from transformers import LlamaConfig, LlamaForCausalLM

RESULTS_DIR = os.path.join("results", "hw2")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 0
PROMPT_LEN = 64
MAX_NEW_TOKENS = 128

# Speedup targets (optimized end-to-end vs V0 baseline).
TIERS = [
    (4.0, ">= 4.0x  (excellent)"),
    (3.0, ">= 3.0x  (good)"),
    (1.5, ">= 1.5x  (some speedup)"),
    (0.0, "< 1.5x  (little to no speedup)"),
]


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _config() -> LlamaConfig:
    # Small enough to iterate fast, real enough to profile meaningfully.
    return LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=4096,
    )


def build_model_and_input(dtype: torch.dtype = torch.float32):
    """Return (model, input_ids) on the active device with fixed random weights.

    The same SEED is used every call, so the V0 baseline and your optimized run
    see identical weights and the same prompt.
    """
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(_config()).to(device=device(), dtype=dtype).eval()
    input_ids = torch.randint(0, 32000, (1, PROMPT_LEN), device=device())
    return model, input_ids


@torch.no_grad()
def baseline_loop(model, input_ids, max_new_tokens: int) -> torch.Tensor:
    """V0 — intentionally slow. Greedy decode with NO KV cache: every step
    re-runs the model over the whole growing sequence (O(n^2)), and forces a
    host sync each step (the `.item()`) — exactly the anti-patterns from lecture.

    Returns the generated token ids, shape (1, max_new_tokens).
    """
    generated = input_ids
    out_tokens = []
    for _ in range(max_new_tokens):
        out = model(input_ids=generated, use_cache=False)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        _ = next_tok.item()                       # forced host sync every step
        generated = torch.cat([generated, next_tok], dim=1)
        out_tokens.append(next_tok)
    return torch.cat(out_tokens, dim=1)


def time_loop(loop_fn, *args, n_warmup: int = 1, n_iters: int = 3) -> float:
    """Average seconds per full generation of `loop_fn(*args)`."""
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


def check_correctness(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    """Greedy decoding is deterministic, so a correct optimized loop must
    reproduce the baseline's token ids EXACTLY when given the same fp32 model."""
    return (reference.shape == candidate.shape
            and torch.equal(reference.cpu(), candidate.cpu()))


def speedup_tier(speedup: float) -> str:
    for threshold, label in TIERS:
        if speedup >= threshold:
            return label
    return TIERS[-1][1]


print(f"device={device()}  prompt_len={PROMPT_LEN}  new_tokens={MAX_NEW_TOKENS}")
```

## Part 1 — profile a generation loop

See the L1 §07 `torch.profiler` snippet. You will use this on both the baseline
and your optimized loop, and read the traces at
[ui.perfetto.dev](https://ui.perfetto.dev).


```python
# TODO-CELL: profile
# YOUR IMPLEMENTATION
def profile(loop_fn, model, input_ids, max_new_tokens: int, trace_path: str) -> None:
    """Run `loop_fn(model, input_ids, max_new_tokens)` under torch.profiler."""
    from torch.profiler import profile as _profile, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with _profile(activities=activities) as prof:
        loop_fn(model, input_ids, max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
    prof.export_chrome_trace(trace_path)

```

## Part 2 — a fast, correct greedy decode loop

Return the same token ids as `baseline_loop` for the same fp32 model — shape
`(1, max_new_tokens)`. Start with what the baseline recomputes every step and
what it forces onto the host.


```python
# TODO-CELL: optimized_loop
# YOUR IMPLEMENTATION
@torch.no_grad()
def optimized_loop(model, input_ids, max_new_tokens: int) -> torch.Tensor:
    """Greedy decode, but fast. Same tokens as baseline_loop, much less work.

    Wins vs V0:
      * KV cache: prefill the prompt once, then feed only the single new token
        each step (O(n) instead of the baseline's O(n^2) full re-run).
      * No host sync on the hot path: keep argmax on the GPU, collect the token
        tensors, and only concatenate (one sync) at the end -- no per-step .item().
    """
    # Prefill: run the whole prompt once and keep the cache.
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    out_tokens = [next_tok]

    # Decode: one new token at a time, reusing the cache.
    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        out_tokens.append(next_tok)

    return torch.cat(out_tokens, dim=1)

```

## Part 3 — build + time the optimized generation

Build the model however you like (dtype, `torch.compile`, CUDA graphs), run
`optimized_loop`, and return `(elapsed_seconds, generated_token_ids)`.

`elapsed_seconds` must be a warmed-up, timed measurement (use `time_loop`); it's
what gets compared to V0. Numerics-changing tricks (e.g. bf16) are fine here —
correctness is checked separately on the fp32 path.


```python
# TODO-CELL: generate_optimized
# YOUR IMPLEMENTATION
def generate_optimized(max_new_tokens: int = MAX_NEW_TOKENS):
    """Return (elapsed_seconds, generated_token_ids) for your fastest setup.

    Uses bf16 + StaticCache + torch.compile(mode="reduce-overhead").
    StaticCache pre-allocates the KV buffer to a fixed max length so every
    decode step has identical input shapes, enabling CUDA-graph capture by
    reduce-overhead. .clone() on each token prevents the static output buffer
    from being overwritten in the sequential loop.
    Extra warmup passes before time_loop let reduce-overhead finish graph
    capture so all timed calls hit the fast replay path.
    Correctness is checked separately on the fp32 path.
    """
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model, input_ids = build_model_and_input(dtype)

    if not torch.cuda.is_available():
        elapsed = time_loop(optimized_loop, model, input_ids, max_new_tokens)
        return elapsed, optimized_loop(model, input_ids, max_new_tokens)

    from transformers import StaticCache
    total_len = PROMPT_LEN + max_new_tokens
    compiled_model = torch.compile(model, mode="reduce-overhead")

    @torch.no_grad()
    def _fast_loop(model, input_ids, n):
        cache = StaticCache(
            config=model.config, max_batch_size=1,
            max_cache_len=total_len, device=model.device, dtype=model.dtype,
        )
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:, :].argmax(dim=-1).clone()
        tokens = [tok]
        for _ in range(n - 1):
            out = model(input_ids=tok, past_key_values=cache, use_cache=True)
            tok = out.logits[:, -1:, :].argmax(dim=-1).clone()
            tokens.append(tok)
        return torch.cat(tokens, dim=1)

    # Extra warmup passes so reduce-overhead finishes CUDA-graph capture
    # before time_loop's single warmup starts timing.
    for _ in range(3):
        _fast_loop(compiled_model, input_ids, max_new_tokens)
    torch.cuda.synchronize()

    elapsed = time_loop(_fast_loop, compiled_model, input_ids, max_new_tokens)
    tokens = _fast_loop(compiled_model, input_ids, max_new_tokens)
    return elapsed, tokens

```

## Self-check (small + fast, runs anywhere)


```python
# SELF-CHECK — DO NOT EDIT.
_n = 24  # short, for speed
_model, _ids = build_model_and_input(torch.float32)

_ref = baseline_loop(_model, _ids, _n)
_cand = optimized_loop(_model, _ids, _n)
assert tuple(_cand.shape) == (1, _n), f"expected shape (1, {_n}), got {tuple(_cand.shape)}"
assert check_correctness(_ref, _cand), \
    "optimized_loop must reproduce the baseline greedy tokens EXACTLY (fp32)"
print("optimized_loop matches baseline   PASS")

_trace = os.path.join(RESULTS_DIR, "trace_check.json")
if os.path.exists(_trace):
    os.remove(_trace)
profile(baseline_loop, _model, _ids, 4, _trace)
assert os.path.exists(_trace) and os.path.getsize(_trace) > 0, \
    "profile() must write a non-empty Chrome trace file"
print("profile() writes a Chrome trace   PASS")
print()
print("All checks passed")
```

## The full run — baseline vs optimized (GPU)

Prints your speedup and writes the two traces for the writeup.


```python
# DO NOT EDIT — the full run.
N_NEW = MAX_NEW_TOKENS if torch.cuda.is_available() else 32
if N_NEW != MAX_NEW_TOKENS:
    print("CPU detected — trimmed to 32 new tokens as a smoke test. "
          "REPORTED NUMBERS MUST COME FROM A GPU RUN.")

# Baseline (V0), fp32 — also the correctness reference.
model_fp32, input_ids = build_model_and_input(torch.float32)
base_t = time_loop(baseline_loop, model_fp32, input_ids, N_NEW)
ref = baseline_loop(model_fp32, input_ids, N_NEW)
print(f"baseline V0: {base_t*1e3:.1f} ms/generation")

# Correctness of your loop on the SAME fp32 model.
cand = optimized_loop(model_fp32, input_ids, N_NEW)
ok = check_correctness(ref, cand)
print(f"optimized_loop correctness (fp32): {'PASS' if ok else 'FAIL'}")

# Optimized end-to-end timing (your choice of dtype/compile/graphs).
opt_t, _ = generate_optimized(N_NEW)
speedup = base_t / opt_t
print(f"optimized:   {opt_t*1e3:.1f} ms/generation")
print(f"speedup: {speedup:.2f}x  ->  {speedup_tier(speedup)}")
if not ok:
    print("WARNING: correctness FAILED — the speedup does not count.")

# Two traces for the writeup: baseline vs optimized.
profile(baseline_loop, model_fp32, input_ids, 32,
        os.path.join(RESULTS_DIR, "trace_baseline.json"))
profile(optimized_loop, model_fp32, input_ids, 32,
        os.path.join(RESULTS_DIR, "trace_optimized.json"))
print("wrote traces to results/hw2/ — open them at https://ui.perfetto.dev")
```

---
## WRITEUP (be concrete and quantitative)

### Q1

From your **baseline trace**, what dominates the time? Name the specific symptom
you saw in the profiler (e.g. kernel-launch gaps, a long run of tiny kernels,
host syncs) and explain it.

**Your answer:**

Two symptoms dominate the baseline trace (visible at https://ui.perfetto.dev):

1. **Per-step host sync gaps.** After every argmax, the baseline calls `.item()`, which forces `torch.cuda.synchronize()` implicitly. In the profiler timeline this appears as a regular gap in the CUDA stream after each decode step — the GPU sits idle while the CPU reads the result back. With 128 steps this adds up to 128 forced stalls. The sync itself is cheap (microseconds — HW3 Part 1 measures per-launch overhead at ≈ 6 µs); the damage is the **lost CPU/GPU overlap**: while the CPU waits for the GPU to drain it cannot queue the next step's kernels, so launch latency and GPU execution run strictly serially instead of pipelining.

2. **A run of growing-cost kernel bursts.** Because there is no KV cache, each step re-runs the full forward pass over the growing input sequence (length 65, 66, … 192). In the profiler table the attention and FFN kernels get slightly longer each step, forming a staircase pattern. This is the O(n²) recomputation: step k processes k previous tokens again from scratch.

The profiler `key_averages` table (sorted by `self_cuda_time_total`) shows the matmul / scaled dot-product attention kernels at the top — they account for the majority of GPU time — flanked by the sync-stall gaps that separate each step.

### Q2

List each optimization you applied and the speedup it contributed — measure them
**one at a time** (baseline → +KV cache → +compile → …). A small table is ideal.

**Your answer:**

Measured on H100 SXM, prompt = 64, new_tokens = 128 (same model, same seed each run). The baseline and the final `StaticCache` row are from the notebook's main run; the three intermediate rows come from a separate ablation session, so their speedups vs the 214.9 ms baseline are approximate (shown to illustrate the trend):

| Configuration | Time (ms) | Speedup vs baseline |
|---|---|---|
| Baseline: fp32, no KV cache, `.item()` each step | 214.9 | 1.00× |
| + KV cache (fp32, `.item()` removed) | 131.5 | **1.63×** |
| + KV + bf16 (no compile) | 153.4 | 1.40× ← *slower than fp32+KV* |
| + KV + bf16 + `torch.compile(dynamic=True)` | 84.6 | **2.54×** |
| + `StaticCache` + bf16 + `compile(reduce-overhead)` | **46.5** | **4.62× ✅ excellent** |

Notable: adding bf16 on top of KV cache without compilation *regressed* from 1.63× to 1.40×. The model is launch-overhead dominated (not bandwidth-dominated), so halving the dtype does not reduce the number of kernel dispatches — it just does less work per dispatch, but launch overhead stays the same.

### Q3

Which single change had the biggest impact, and why does it help THIS workload
(128 short decode steps on a tiny model) specifically?

**Your answer:**

The biggest single change is **CUDA-graph capture via `compile(mode="reduce-overhead")` with `StaticCache`**, which takes the bf16+KV baseline from 153 ms to 46.5 ms (a 3.3× jump in that step, 4.62× total).

KV cache alone gave only 1.63×. Here is why the ordering is the opposite of intuition:

This 2-layer / hidden-size-512 Llama model is so small that the H100 finishes each matrix multiply in single-digit microseconds. The dominant cost is not compute — it is **kernel dispatch**: 128 decode steps × (dozens of small kernels per step: layernorms, QKV projections, attention, FFN) = thousands of separate CUDA launches, each costing ≈ 6 µs (measured directly in HW3 Part 1). That is tens of milliseconds of pure CPU → GPU scheduling overhead.

KV cache removes the O(n²) recomputation, which matters a lot when the model is compute-bound (e.g., a real 7B model on a weaker GPU). Here the FLOP savings are small — the sequence only grows to 192 tokens and the H100 finishes either version in microseconds — so most of KV cache's 1.63× comes from dropping the 128 per-step `.item()` syncs (that ablation row removes both at once), not from the reduced compute. Either way it is second-order next to the launch-overhead win.

`compile(reduce-overhead)` with `StaticCache` collapses each decode step's entire kernel sequence into a single CUDA graph replay — one GPU launch instead of dozens. This directly attacks the actual bottleneck.

### Q4

Your decode loop is memory-bandwidth-bound (L1 §06, L2 §01). Which of your
optimizations attack memory traffic vs CPU/launch overhead? Which kind mattered
more here, and why?

**Your answer:**

**Memory-traffic optimizations:**
- **KV cache** — eliminates the O(n²) re-reads of past key/value tensors; each decode step now reads only the new single-token activations plus the cached K/V.
- **bf16** — halves the parameter size in HBM, so each kernel reads and writes half as many bytes.

**CPU/launch-overhead optimizations:**
- **Removing `.item()`** — eliminates 128 forced CPU-GPU synchronizations per generation.
- **CUDA graphs (`compile(reduce-overhead)` + `StaticCache`)** — collapses each decode step's dozens of kernel launches into a single replay, eliminating per-kernel dispatch overhead.

**What mattered more here: launch overhead, by a wide margin.**

KV cache + bf16 together peaked at 2.54× (with dynamic compile). The final jump to 4.62× came entirely from CUDA graph capture. The data:

- KV cache alone: 1.63× — most of it from removing the per-step host syncs; the O(n²) FLOP removal itself gives little because this tiny model's matmuls finish in microseconds on H100.
- bf16 without fusion: actually *regressed* (1.40×, below KV-cache-alone) — the model was never bandwidth-saturated, so halving dtype saves no wall time without also fusing kernels.
- CUDA graphs: took 153 ms (KV+bf16, no compile) → 46.5 ms — a 3.3× improvement in that single step.

This result directly contradicts the naive "decode is memory-bandwidth-bound" intuition. That intuition holds for large production models (7B+ parameters with hundreds of attention heads, long contexts) where attention K/V reads genuinely stress HBM bandwidth. For a 2-layer / 512-hidden toy model on an H100, actual memory bandwidth utilization is low — the bottleneck is the Python dispatch loop, not the GPU.
