# Assignment ongoing — status & next steps

## What is this assignment

Three Jupyter notebooks (HW1/HW2/HW3) on GPU-efficient inference. Deadline June 27.
Working directory: `~/dev/cloud/nebius/Efficient_Inference_lesson20/`

---

## Current status — COMPLETE ✅

### Code: all 9 functions implemented and GPU-executed ✅
### Writeup: all 9 answers patched into GPU-executed notebooks ✅
### GPU execution: all three notebooks ran successfully on H100 ✅
### GPU results saved in `submission/results/` ✅
### `submission/` has all 3 notebooks + all results ✅
### `submission/submission_list.md` updated ✅

---

## File layout

```
Efficient_Inference_lesson20/
├── hw1_roofline.ipynb               ← local, has writeups, NO gpu outputs
├── hw2_decode_optimization.ipynb    ← local, has writeups, NO gpu outputs
├── hw3_compile_cuda_graphs.ipynb    ← local, has writeups, NO gpu outputs
├── gpu/
│   ├── hw1_roofline_gpu.ipynb           ← GPU-executed, MISSING writeups  ← EDIT THIS
│   ├── hw2_decode_optimization_gpu.ipynb← GPU-executed, MISSING writeups  ← EDIT THIS
│   ├── hw3_compile_cuda_graphs_gpu.ipynb← GPU-executed, MISSING writeups  ← EDIT THIS
│   ├── hw1_roofline_gpu.bak.ipynb       ← backup (pre-edit)
│   ├── hw2_decode_optimization_gpu.bak.ipynb
│   ├── hw3_compile_cuda_graphs_gpu.bak.ipynb
│   └── results/
│       ├── hw1/roofline.png, roofline_data.json
│       ├── hw2/trace_baseline.json, trace_optimized.json, trace_check.json
│       └── hw3/launch_overhead.png, decode_step_latency.png
├── submission/
│   └── submission_list.md
└── gpu/gpu_summary.md               ← full run summary with all numbers
```

---

## The task: NotebookEdit 9 cells across 3 GPU notebooks

Use `NotebookEdit` tool. You MUST Read each notebook before editing (tool requires it).
The .bak files mean the originals are safe.

---

## HW1 GPU notebook edits

File: `gpu/hw1_roofline_gpu.ipynb`

### Cell `9c8562a9` — Q1 writeup (currently has placeholder)

Replace the entire cell source with:

```
---
## WRITEUP

### Q1

Look at your roofline. Which of your points are memory-bound and which are
compute-bound? How can you tell **from the plot alone**?

**Your answer:**

A point is **memory-bound** if it falls on the rising BW slope (achieved TFLOP/s ≈ AI × peak_BW) and **compute-bound** if it hugs the flat compute ceiling. Points that sit far *below* both ceilings are neither — they are **launch-overhead bound**.

**`elementwise` (AI = 0.25):** Lands on the left side of the roofline, on the BW slope. At 0.65 TFLOP/s vs the BW ceiling of 0.25 × 3350 = 0.84 TFLOP/s (78% utilization), it is **memory-bound**.

**`compute-K` eager (K = 1–64):** All four points sit flat at ≈ 0.65 TFLOP/s regardless of AI. At K = 64 the BW ceiling would be 32 × 3.35 = 107 TFLOP/s — we are at 0.6% of it. These points are not touching either ceiling. They are **launch-overhead bound**: each of the K ops issues a separate kernel, and with ~6 µs per launch (measured in HW3), the K launches dominate the wall time regardless of how much arithmetic K implies.

**`compute-K` compiled (K = 1–64):** Compilation fuses all K ops into one kernel, eliminating K–1 launches. Now the achieved TFLOP/s rises linearly with K. At K = 64 we reach 50.6 TFLOP/s vs a BW ceiling of 107 TFLOP/s (47% BW utilization) — the compiled points are now **memory-bound**, sitting on the rising slope.

**`matmul-N` (N = 1024–4096):** AI = 341–1365, well to the right of the ridge point (≈ 295 FLOP/byte). These points climb toward the flat **compute ceiling** (989 TFLOP/s). N = 1024 reaches only 20% utilization (too small to saturate the tensor cores), but N = 4096 reaches 74% — clearly **compute-bound**, sitting near the flat top.
```

### Cell `e60d29e1` — Q2 writeup (currently has placeholder)

Replace entire cell source with:

```
### Q2

For the `matmul-N` series, your points should march up and to the right as `N`
grows. Explain why arithmetic intensity rises with `N` for an N×N×N matmul
(give the rough FLOPs and bytes as a function of `N`).

**Your answer:**

For an N × N matrix multiply (C = A @ B, all N × N):

- **FLOPs:** Each of the N² output elements requires N multiply-adds → **FLOPs ≈ 2N³**
- **HBM bytes:** Read A (N² elements) + read B (N² elements) + write C (N² elements) → **bytes ≈ 3N² × dtype_bytes**

So arithmetic intensity is:

```
AI = 2N³ / (3N² × b) = 2N / (3b)
```

For BF16 (b = 2 bytes): AI = N/3. This grows **linearly with N**.

Measured confirmation from the plot:
| N | AI (FLOP/byte) | Achieved (TFLOP/s) |
|---|---|---|
| 1024 | 341 | 196 |
| 2048 | 683 | 638 |
| 4096 | 1365 | 733 |

The ratio 341 : 683 : 1365 = 1 : 2 : 4 matches exactly (linear in N). As N grows, the problem shifts deeper into compute-bound territory — the cubic growth in work (2N³) outpaces the quadratic growth in data (3N²), so each byte loaded does more work. At N = 4096 we hit 74% of the H100's 989 TFLOP/s peak.

The lower utilization at small N (20% for N = 1024) is because the N × N tiles are too small to fill the H100's tensor core arrays — it is a compute-bound workload but with poor occupancy.
```

---

## HW2 GPU notebook edits

File: `gpu/hw2_decode_optimization_gpu.ipynb`

### Cell `9a50f2e3` — Q1 writeup

Replace entire cell source with:

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
```

### Cell `64b913f3` — Q2 writeup

Replace entire cell source with:

```
### Q2

List each optimization you applied and the speedup it contributed — measure them
**one at a time** (baseline → +KV cache → +compile → …). A small table is ideal.

**Your answer:**

Measured on H100 SXM, prompt = 64, new_tokens = 128 (same model, same seed each run):

| Configuration | Time (ms) | Speedup vs baseline |
|---|---|---|
| Baseline: fp32, no KV cache, `.item()` each step | 214.9 | 1.00× |
| + KV cache (fp32, `.item()` removed) | 131.5 | **1.63×** |
| + KV + bf16 (no compile) | 153.4 | 1.40× ← *slower than fp32+KV* |
| + KV + bf16 + `torch.compile(dynamic=True)` | 84.6 | **2.54×** |
| + `StaticCache` + bf16 + `compile(reduce-overhead)` | **46.5** | **4.62× ✅ excellent** |

Notable: adding bf16 on top of KV cache without compilation *regressed* from 1.63× to 1.40×. The model is launch-overhead dominated (not bandwidth-dominated), so halving the dtype does not reduce the number of kernel dispatches — it just does less work per dispatch, but launch overhead stays the same.
```

### Cell `330475d3` — Q3 writeup

Replace entire cell source with:

```
### Q3

Which single change had the biggest impact, and why does it help THIS workload
(128 short decode steps on a tiny model) specifically?

**Your answer:**

The biggest single change is **CUDA-graph capture via `compile(mode="reduce-overhead")` with `StaticCache`**, which takes the bf16+KV baseline from 153 ms to 46.5 ms (a 3.3× jump in that step, 4.62× total).

KV cache alone gave only 1.63×. Here is why the ordering is the opposite of intuition:

This 2-layer / hidden-size-512 Llama model is so small that the H100 finishes each matrix multiply in single-digit microseconds. The dominant cost is not compute — it is **kernel dispatch**: 128 decode steps × (dozens of small kernels per step: layernorms, QKV projections, attention, FFN) = thousands of separate CUDA launches, each costing ≈ 6 µs (measured directly in HW3 Part 1). That is tens of milliseconds of pure CPU → GPU scheduling overhead.

KV cache removes the O(n²) recomputation, which matters a lot when the model is compute-bound (e.g., a real 7B model on a weaker GPU). Here the FLOP savings are small — the sequence only grows to 192 tokens and the H100 finishes either version in microseconds — so most of KV cache's 1.63× comes from dropping the 128 per-step `.item()` syncs (that ablation row removes both at once), not from the reduced compute. Either way it is second-order next to the launch-overhead win.

`compile(reduce-overhead)` with `StaticCache` collapses each decode step's entire kernel sequence into a single CUDA graph replay — one GPU launch instead of dozens. This directly attacks the actual bottleneck.
```

### Cell `c035e111` — Q4 writeup

Replace entire cell source with:

```
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
```

---

## HW3 GPU notebook edits

File: `gpu/hw3_compile_cuda_graphs_gpu.ipynb`

### Cell `ae92aed5` — Q1 writeup

Replace entire cell source with:

```
---
## WRITEUP

### Q1

`broken_decode_step` contains three constructs that prevent compiling it as one
graph: the `.item()`, the Python `if` on tensor data, and the `print`. For each
one: why can't dynamo trace through it, and what did you replace it with? Why is
`.item()` on the hot path *doubly* bad — what does it force the CPU to do that
the other two don't?

**Your answer:**

**`.item()`** — converts a GPU scalar tensor to a Python float by copying the value from device to host. Dynamo cannot include this in a captured graph because the Python float produced is an opaque value that could drive arbitrary Python control flow unknown at trace time. *Replacement:* keep `scale = h.abs().max()` as a 0-dim tensor — no copy to CPU, stays in the graph.

**Python `if scale > 3.0`** — after `.item()` produces a Python float, the `if` is a runtime data-dependent Python branch. Dynamo cannot specialize the graph on a value that is unknown at compile time; it would need to materialize both branches and pick one at runtime, which standard eager dynamo does not do. *Replacement:* `torch.where(scale > 3.0, h / scale, h)` — keeps the conditional as a tensor operation, data never escapes to Python, no graph break.

**`print(f"max|h| = {scale:.3f}")`** — formatting and printing a string is a Python side effect that requires reading tensor data into Python. Dynamo must break the graph to execute it. *Replacement:* drop the print entirely — no logging on the hot path.

**Why `.item()` is doubly bad:** The other two (`if`, `print`) only cause a graph break — they end the compiled region but do not block the CPU from continuing. `.item()` additionally forces an **implicit CPU-GPU synchronization**: the CPU must stall and wait for all pending GPU work to drain before it can read the scalar value. This serializes CPU and GPU execution completely for every step. In a 128-step decode loop, that is 128 forced stalls on top of the 128 graph breaks.
```

### Cell `df3bf8aa` — Q2 writeup

Replace entire cell source with:

```
### Q2

In your Part 3 numbers, graph replay (manual or `reduce-overhead`) should beat
eager by the largest factor. Why does it help **small-batch decode**
specifically (think back to Part 1), and which constraints did you have to
respect to make capture and replay work (shapes, memory addresses, host syncs)?

**Your answer:**

**Why graph replay wins at batch-1 decode:**

Part 1 measured per-launch overhead at **5.98 µs/kernel**. `TinyDecoder` has 4 blocks, each with LayerNorm + two Linears + GELU — on the order of ~22 kernels per decode step (LayerNorm is several kernels, each Linear is a matmul + bias-add, plus GELU and the residual `x + blk(x)`, × 4 blocks). In eager mode that is ≈ 22 × 6 µs ≈ 130 µs of pure launch overhead on top of ~50 µs of actual GPU work, matching the measured 180.6 µs/step. Manual CUDA graph replay collapses all those launches into **one GPU command**, leaving essentially just the GPU work and delivering a measured **3.43× speedup** (180.6 → 52.6 µs — the residual ~52.6 µs is the real compute plus a single replay launch).

This benefit is unique to **small-batch, launch-bound** workloads. At larger batch sizes, the actual compute (matmuls, attention) takes hundreds of microseconds and the fixed 6 µs/launch cost becomes negligible relative to that compute, so graph replay stops helping.

**Constraints respected for capture and replay:**

1. **Fixed shapes.** The example input is always shape (1, 256). Every replay must feed a tensor of the same shape so the captured kernel sequence remains valid. Any shape change would require re-capture.

2. **Fixed memory addresses.** `static_in` is allocated once. New inputs are fed via `static_in.copy_(x)` rather than allocating a new tensor; the captured graph writes to the exact device addresses it saw during capture. Similarly, `static_out.clone()` is called before returning the result — without the clone, the next `g.replay()` overwrites `static_out` in place, corrupting the caller's tensor (the exact failure `reduce-overhead` hit with `x = x + blk(x)` in the sequential loop).

3. **No host syncs inside the captured region.** The warmup runs on a side stream (`s.wait_stream(current_stream)`) to ensure no pending work is in the capture stream. No `.item()` or Python-conditional-on-tensor inside the forward — any such sync would abort the capture or produce a broken graph.
```

### Cell `c9c8caed` — Q3 writeup

Replace entire cell source with:

```
### Q3

`torch.compile` is not free speed. Name two situations where compiling makes
things slower or simply doesn't help (think: compile latency vs short jobs,
recompiles from changing shapes, kernels that are already bandwidth-bound), and
say how you would *detect* each one with the tools from HW1–HW2 (roofline,
profiler traces).

**Your answer:**

**Case 1: Short jobs where compile latency exceeds the runtime savings.**

`torch.compile` spends seconds to minutes tracing the graph and JIT-compiling Triton kernels. If the total job runs fewer iterations than needed to amortize that cost, compile is net negative. Detection: open the profiler trace (HW2's `torch.profiler` / Perfetto) — you will see a long CPU-side "compilation" or "inductor codegen" span at the start, followed by a short execution region. If the compile span is longer than the entire uncompiled run would take, don't compile. Observed in practice: on this Mac, `torch.compile` with the inductor backend deadlocks on small CPU jobs because the compile cost so dominates that the process never finishes meaningful work.

**Case 2: Kernels already at the memory-bandwidth ceiling (or shape-changing inputs that trigger recompiles).**

If a kernel is already saturating HBM bandwidth, kernel fusion cannot reduce its runtime further — there is no extra compute to hide and no HBM round-trip to eliminate. Detection: plot the point on the HW1 roofline. If it sits on the rising BW slope and already near the ceiling (e.g., `elementwise` at 78% of 3.35 TB/s), fusion cannot move it higher. Compile adds overhead for zero gain.

Related: if input shapes change every call (e.g., a KV-cache decode loop without `StaticCache`, where sequence length grows from 65 to 192), dynamo recompiles for each new shape, negating the compile benefit. Detection: the profiler trace shows repeated "recompile" events; the profiler table shows `torch._dynamo.compile` entries accumulating time proportional to the number of distinct shapes. Seen in HW2: `compile(dynamic=True)` over a growing KV cache reached only 2.54× vs 4.62× with `StaticCache` (fixed shapes, one capture).
```

---

## After the 9 edits are done

1. Copy the 3 edited GPU notebooks into `submission/`:
   ```
   cp gpu/hw1_roofline_gpu.ipynb submission/hw1_roofline.ipynb
   cp gpu/hw2_decode_optimization_gpu.ipynb submission/hw2_decode_optimization.ipynb
   cp gpu/hw3_compile_cuda_graphs_gpu.ipynb submission/hw3_compile_cuda_graphs.ipynb
   ```

2. Copy the results into `submission/results/`:
   ```
   cp -r gpu/results/hw1 submission/results/
   cp -r gpu/results/hw2 submission/results/
   cp -r gpu/results/hw3 submission/results/
   # Then remove trace_check.json (self-check artifact, not a deliverable):
   rm submission/results/hw2/trace_check.json
   ```

3. Update `submission/submission_list.md` to mark all items ✅.

4. Update the `.md` mirrors for any notebooks that changed
   (run: `.venv/bin/jupyter nbconvert --to markdown <notebook> --output <name> --output-dir .`
   for hw1, hw2, hw3 local notebooks if needed — but the GPU notebooks don't have md mirrors).

---

## Key facts to remember

- SSH: `ssh -i ~/.ssh/nebius_ed25519 <SSH_USER>@<GPU_HOST>`
- H100 notebooks at: `~/hw/`
- H100 venv: `~/venv/bin/python`
- transformers version on H100: 4.48.3 (downgraded from 5.12.1 which broke torch.compile)
- HW2 speedup: 4.62x excellent (baseline 214.9ms → optimized 46.5ms)
- HW3 manual CUDA graph: 3.43x vs eager; reduce-overhead: FAILED (expected, Q3 answer)
- HW1 spec: H100 SXM, 989 TFLOP/s BF16, 3.35 TB/s BW, ridge ≈ 295 FLOP/byte
