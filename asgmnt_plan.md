# Efficient Inference (Lesson 20) — Assignment Plan

**Deadline:** June 27 (also on Google Classroom).
**Deliverables:** the three executed notebooks + everything they write under `results/`.

Three notebooks, all on the same theme (the GPU roofline → why decode is
launch/bandwidth-bound → how to fix it):

| HW | Topic | What you fill in |
|----|-------|------------------|
| `hw1_roofline.ipynb` | Roofline & arithmetic intensity | `lowest_ai_fn`, `make_compute_fn`, `benchmark_fn`, `compute_metrics`, writeup Q1–Q2 |
| `hw2_decode_optimization.ipynb` | Profile + speed up a greedy decode loop | `profile`, `optimized_loop`, `generate_optimized`, writeup Q1–Q4 |
| `hw3_compile_cuda_graphs.ipynb` | `torch.compile` & CUDA graphs | `sweep_elementwise_times`+`estimate_launch_overhead_us`, `fixed_decode_step`, `make_graphed_callable`, writeup Q1–Q3 |

---

## Where to run it — Mac vs Colab vs Nebius

**Short version: do NOT use the Mac for the graded run. Use a Nebius GPU (L40S or
H100). Colab is an acceptable fallback. Use the Mac only to write/debug code.**

Why the M4 Pro is a dead end for submission:
- Every notebook branches on `torch.cuda.is_available()`. On macOS that is
  `False` (PyTorch's Apple backend is **MPS/Metal**, not CUDA), so you only ever
  hit the CPU-fallback paths.
- HW1 prints *"Results are NOT meaningful; run on H100/L40S."*
- HW2: *"the speedup only counts if … the timed run is on a GPU."*
- HW3 Part 3 (`make_graphed_callable`) and the final benchmark **`SKIP` on CPU** —
  `torch.cuda.CUDAGraph` / `Stream` / events simply don't exist without CUDA.
  This is the centerpiece of HW3 and cannot run on the Mac at all.

Why **Nebius L40S/H100** is the best target:
- HW1's `detect_hardware()` only recognizes `H100` and `L40` device names and
  loads the correct BF16/HBM peak ceilings for them. Any other card (e.g. a Colab
  **T4**) silently *"Defaults to H100 ceilings"* → your measured points sit far
  below a roof that isn't yours, so the roofline is miscalibrated and Q1/Q2 get
  awkward. L40S/H100 give a correct roof out of the box.
- HW3 literally says *"Run this notebook on your Nebius instance."*
- You already have Nebius access as a student — no quota lottery.

Colab notes if you go that route anyway:
- Free tier = **T4**: fine for HW2/HW3 (they only need *ratios* — speedups), but
  HW1's roofline ceilings will be wrong (T4 → H100 default). If you must use a T4
  for HW1, add a T4 `GpuSpec` (peak BF16 ≈ 65 TFLOP/s, BW ≈ 0.30 TB/s) — but note
  the setup cell is marked **DO NOT EDIT**, so prefer asking the instructor to add
  the spec, or just use L40S/H100.
- Colab Pro can give L4/A100 — also not in the recognized list, same caveat.
- `torch.compile` + CUDA graphs need a recent PyTorch + CUDA; Colab/Nebius images
  generally have this. The notebooks already wrap `torch.compile` in try/except
  fallbacks, but you want it actually working for HW2/HW3 to hit the good tiers.

**Plan: write & smoke-test the logic on the Mac (CPU paths + self-checks pass
anywhere), then do the real, graded execution on a Nebius L40S or H100.**

---

## Environment setup (on the GPU box)

```bash
# Nebius GPU instance (L40S or H100), or Colab with GPU runtime
pip install torch transformers matplotlib numpy
# HW2 needs transformers (LlamaConfig/LlamaForCausalLM); requirements.txt only — no vLLM/TRT-LLM/SGLang
```
Run each notebook **top-to-bottom**. Each creates `results/hw1|hw2|hw3/` itself.

---

## HW1 — Roofline

Concepts: `achievable = min(peak_compute, peak_BW × AI)`; AI = FLOPs / HBM bytes.

1. **`lowest_ai_fn(x)`** — return a *no-arg* closure doing one cheap elementwise
   pass, e.g. `return lambda: x * 2.0`. ~1 FLOP/elem, 1 read + 1 write → lands
   memory-bound (far left). One read+one write is the convention the run cell
   already assumes (`bytes_moved = 2*N*dbytes`).
2. **`make_compute_fn(x, k)`** — return a no-arg closure that chains `k`
   elementwise ops, each feeding the next, **without mutating `x`**. Real data
   dependency so steps can't be constant-folded, e.g.
   ```python
   def make_compute_fn(x, k):
       def run():
           y = x.clone()           # don't mutate x
           for _ in range(k):
               y = y * 1.0001 + 0.5   # mul+add = 2 FLOP/elem/step (matches flops=2*k*N)
           return y
       return run
   ```
   Eager: each step is its own kernel + HBM round-trip → low achieved AI.
   The run cell also wraps it in `torch.compile`, which fuses the chain → point
   moves up toward the compute roof. (Make the per-step math match `2*k*N` FLOPs.)
3. **`benchmark_fn(fn, n_warmup, n_iters)`** — warm up untimed, then time with
   `torch.cuda.Event(enable_timing=True)` + `torch.cuda.synchronize()` before
   reading; CPU fallback uses `time.perf_counter`. Return **seconds/call**
   (`elapsed_total / n_iters`).
4. **`compute_metrics(flops, bytes_moved, seconds)`** — return exactly
   ```python
   {"ai": flops/bytes_moved,
    "achieved_tflops": flops/seconds/1e12,
    "achieved_gbps": bytes_moved/seconds/1e9}
   ```
   (Self-check: flops=2e12, bytes=1e9, s=1e-3 → ai=2000, 2000 TFLOP/s, 1000 GB/s.)
5. Run the sweep cell → writes `results/hw1/roofline_data.json` + `roofline.png`.
   Sanity: all points under the roof; `elementwise`/small-K low-left; `matmul-N`
   climbs up-right; compiled compute-K above eager at same AI.
6. **Writeup:** Q1 — points left of the ridge (and rising with the BW slope) are
   memory-bound, points hugging the flat top are compute-bound; you tell from the
   plot by which ceiling each sits under. Q2 — N×N×N matmul: FLOPs ≈ 2N³, HBM
   bytes ≈ 3N²·dbytes → AI ≈ (2N³)/(3N²·b) ∝ N, so AI grows linearly with N.

## HW2 — Profile & optimize greedy decode

Baseline V0 = no KV cache (recomputes whole O(n²) sequence), fp32 eager, `.item()`
host-sync every step. Target ≥4× (excellent), ≥3× good, ≥1.5× some.

Rules: PyTorch only (no vLLM/TRT-LLM/SGLang); don't edit harness; `optimized_loop`
must reproduce baseline greedy tokens **exactly on the same fp32 model**;
`generate_optimized` may use bf16.

1. **`profile(...)`** — wrap the loop in `torch.profiler.profile` with
   `ProfilerActivity.CPU` + `.CUDA`, print `key_averages().table(sort_by=
   "self_cuda_time_total")` (CPU sort on CPU-only), and
   `prof.export_chrome_trace(trace_path)`.
2. **`optimized_loop(...)`** — the big wins, in order:
   - **KV cache**: prefill the prompt once with `use_cache=True`, then feed only
     the *single new token* each step with `past_key_values` (turns O(n²) into
     O(n)). This is the biggest lever and alone should clear the correctness bar.
   - **Kill the host sync**: don't `.item()` on the hot path; keep `argmax` on GPU,
     accumulate tokens in a list/tensor, sync once at the end.
   - Must stay fp32 here to match tokens exactly (the timed run can differ).
3. **`generate_optimized(...)`** — build your fastest setup (KV cache + bf16, and
   optionally `torch.compile` / `cache_implementation="static"`), warm up, time
   with `time_loop`, return `(elapsed_seconds, token_ids)`. bf16 is allowed here
   because correctness is checked separately on the fp32 path.
4. Run the full cell → speedup + `results/hw2/trace_baseline.json` &
   `trace_optimized.json` (open at https://ui.perfetto.dev).
5. **Writeup:** Q1 baseline trace symptom (long run of growing-matmul + a host sync
   every step from `.item()`); Q2 ablation table measured **one at a time**
   (baseline → +KV → +bf16 → +compile); Q3 biggest single change = **KV cache**
   (kills the O(n²) recompute that dominates 128 short steps); Q4 which opts cut
   *memory traffic* (KV cache, bf16) vs *launch/CPU overhead* (compile, no sync) —
   for this tiny bandwidth-bound model the memory-traffic ones dominate.

## HW3 — `torch.compile` & CUDA graphs

Parts 1–2 run anywhere; **Part 3 + final benchmark need the GPU.**

1. **`sweep_elementwise_times(sizes)`** — for each `n`, build a 1-D fp32 tensor on
   `DEVICE`, time **one** out-of-place op (e.g. `x + 1.0`) with `bench()`, same op
   for every size. **`estimate_launch_overhead_us`** — average seconds/call over
   the launch-bound sizes (`n <= 2**12`), ×1e6 → µs. Run cell → `launch_overhead.png`.
2. **`fixed_decode_step(h, state)`** — same math as `broken_decode_step` minus the
   print, **zero graph breaks** (must pass `fullgraph=True`):
   - keep `scale = h.abs().max()` as a **0-dim tensor** (no `.item()`);
   - replace the Python `if scale > 3.0` with `torch.where(scale > 3.0, h/scale, h)`;
   - drop the `print`.
   Self-check verifies identical numerics across cases + 0 graph breaks via
   `dynamo.explain`.
3. **`make_graphed_callable(fn, example_input)`** — manual CUDA-graph capture:
   1) `static_in = example_input.clone()`; 2) warm up on a **side stream**
   (`s.wait_stream(current)`, run a few iters under `torch.cuda.stream(s)`,
   `current.wait_stream(s)`); 3) capture once under `torch.cuda.graph(g)` →
   `static_out = fn(static_in)`; 4) `graphed(x)`: `static_in.copy_(x)`,
   `g.replay()`, `return static_out.clone()`. The `copy_` + `clone` are what the
   self-check's "different inputs → different outputs" assert catches.
4. Final benchmark cell compares eager / manual graph / `compile(default)` /
   `compile(reduce-overhead)` → `decode_step_latency.png`. (If `reduce-overhead`
   errors about overwritten outputs in the sequential loop, that's the
   static-buffer constraint — explain in Q3, don't work around it.)
5. **Writeup:** Q1 why `.item()`/`if`/`print` break dynamo and your replacements;
   `.item()` is *doubly* bad — graph break **and** forces a CPU↔GPU sync (CPU
   stalls). Q2 graph replay wins at batch-1 because the step is launch-bound
   (Part 1 overhead) and replay turns N launches into 1; constraints = fixed
   shapes, fixed memory addresses (static buffers), no host syncs. Q3 two cases
   where compile doesn't pay: short jobs (compile latency > runtime) and
   shape-changing inputs (recompiles) / already-bandwidth-bound kernels — detect
   via profiler traces (compile/recompile gaps) and the HW1 roofline (already at
   the BW ceiling → no room).

---

## Suggested order of work

1. **On the Mac:** implement all functions; run each notebook's **SELF-CHECK**
   cells (they pass on CPU) to validate logic. Iterate fast here.
2. **On Nebius (L40S/H100):** run all three notebooks top-to-bottom for the real
   numbers, traces, and plots. Confirm HW2 hits a good speedup tier and HW3 Part 3
   doesn't SKIP.
3. Fill in the four writeup sections using the actual plots/traces.
4. Submit the executed notebooks + the `results/` files.
