# GPU Run Summary — H100 SXM, June 19 2026

**Hardware:** NVIDIA H100 80GB HBM3 (H100 SXM)  
**Environment:** Python 3.12, torch 2.5.1+cu121, transformers 4.48.3, CUDA driver 580  
**Files:** `gpu/hw{1,2,3}_*_gpu.ipynb` + `gpu/results/hw{1,2,3}/`

---

## HW1 — Roofline

**Spec (from `detect_hardware`):** peak BF16 = 989 TFLOP/s, peak BW = 3.35 TB/s  
Ridge point AI = 989/3350 ≈ 0.295 FLOP/byte

| Label | Series | AI (FLOP/byte) | Achieved (TFLOP/s) | Region |
|---|---|---|---|---|
| elementwise | memory-bound | 0.25 | 0.65 | memory-bound (78% BW util) |
| K=1 | compute-K eager | 0.50 | 0.65 | launch-overhead bound |
| K=4 | compute-K eager | 2.0 | 0.65 | launch-overhead bound |
| K=16 | compute-K eager | 8.0 | 0.65 | launch-overhead bound |
| K=64 | compute-K eager | 32.0 | 0.65 | launch-overhead bound |
| K=1 | compute-K compiled | 0.50 | 1.31 | memory-bound (fused) |
| K=4 | compute-K compiled | 2.0 | 5.23 | memory-bound (fused) |
| K=16 | compute-K compiled | 8.0 | 21.1 | memory-bound (fused) |
| K=64 | compute-K compiled | 32.0 | 50.6 | memory-bound (fused, ~47% BW) |
| matmul N=1024 | matmul-N | 341 | 196 | compute-bound (~20% util — too small) |
| matmul N=2048 | matmul-N | 683 | 638 | compute-bound (~65% util) |
| matmul N=4096 | matmul-N | 1365 | 733 | compute-bound (~74% util) |

Key observations:
- Eager compute-K runs are completely flat at 0.65 TFLOP/s regardless of K. The bottleneck is kernel launch overhead (~6 µs/launch × K launches), not memory or compute.
- Compiling fuses all K ops into one kernel: achieved TFLOP/s rises linearly with K, now limited by BW.
- Matmul climbs the compute ceiling as N grows. N=4096 reaches 74% of peak — compute-bound.

---

## HW2 — Decode Optimization

**Model:** 2-layer Llama (hidden=512, ffn=1376, heads=8), prompt=64, new_tokens=128

### Ablation (standalone measurement, same model structure)

| Optimization | Time (ms) | Speedup |
|---|---|---|
| Baseline (fp32, no-KV, `.item()` each step) | 214.9 | 1.00x |
| + KV cache (fp32, no `.item()`) | 131.5 | **1.63x** |
| + KV + bf16 (no compile) | 153.4 | 1.40x ← slower than fp32+KV! |
| + KV + bf16 + `compile(dynamic)` | 84.6 | **2.54x** |
| + StaticCache + bf16 + `compile(reduce-overhead)` | **46.5** | **4.62x ✅ excellent** |

> Speedups for the 3 middle rows are recomputed vs the 214.9 ms baseline; their ms values come from this standalone run (separate session), so they're approximate. The 214.9 / 46.5 / 4.62× anchors are from the notebook's main run and are exact.

### What the numbers reveal

KV cache removes the O(n²) recomputation but this model is so small (2 layers, 512 hidden) that the H100 finishes each sequence in microseconds. The real bottleneck is **kernel launch overhead**: 128 steps × dozens of kernels × ~6 µs/launch = ~tens of ms of pure dispatch cost. bf16 alone made things *slower* without fusion because halving dtype doesn't help when you're not bandwidth-limited — launch overhead is unchanged. Only `torch.compile` with CUDA-graph capture (`reduce-overhead`) actually kills the launch overhead: it collapses 128 × N launches into batched graph replay.

The `.item()` removal (part of the +KV step) helps by restoring CPU/GPU overlap, not by the sync's own cost (≈6 µs/launch, measured in HW3): with the per-step sync gone, the CPU can queue the next step's kernels instead of stalling for the GPU to drain.

### Traces
- `results/hw2/trace_baseline.json` — open at https://ui.perfetto.dev
- `results/hw2/trace_optimized.json`

---

## HW3 — torch.compile & CUDA Graphs

### Part 1 — kernel launch overhead

| n (elements) | time/call |
|---|---|
| 1 – 1,048,576 | ~5.75–6.28 µs (flat — launch-bound) |
| 4,194,304 | 8.80 µs (knee — compute starts) |
| 16,777,216 | 47.57 µs (compute-bound) |

**Estimated per-launch overhead: 5.98 µs**

### Part 2 — fixed_decode_step

| | |
|---|---|
| `same math as the broken step` | PASS |
| `compiles with fullgraph=True` | PASS |
| `0 graph breaks` | PASS |
| broken (eager, sync+print) | 75.9 µs/step |
| **fixed (compiled, one graph)** | **34.2 µs/step (2.2×)** |

### Part 3 — make_graphed_callable

| | |
|---|---|
| `matches eager on fresh inputs` | PASS |
| `static buffers handled correctly` | PASS |

### Final benchmark

| Method | Latency | vs eager |
|---|---|---|
| eager | 180.6 µs/step | 1.00× |
| **manual CUDA graph** | **52.6 µs/step** | **3.43×** |
| `compile(default)` | 178.5 µs/step | 1.01× |
| `compile(reduce-overhead)` | FAILED | — |

`reduce-overhead` fails with: *"accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run"* — the `x + blk(x)` residual reuses the same static output buffer across sequential steps, which CUDA graphs can't handle without `clone()` or `cudagraph_mark_step_begin()`. This is the Q3 teaching point.

`compile(default)` matches eager (1.01×) because this tiny step is launch-overhead dominated — kernel fusion alone doesn't help when there's no compute to fuse across, and default mode doesn't capture CUDA graphs.

---

## File manifest

```
gpu/
├── gpu_summary.md                    ← this file
├── hw1_roofline_gpu.ipynb            ← executed, H100 results
├── hw2_decode_optimization_gpu.ipynb ← executed, 4.62x excellent
├── hw3_compile_cuda_graphs_gpu.ipynb ← executed, Part 3 PASS
└── results/
    ├── hw1/
    │   ├── roofline.png
    │   └── roofline_data.json
    ├── hw2/
    │   ├── trace_baseline.json
    │   └── trace_optimized.json
    └── hw3/
        ├── launch_overhead.png
        └── decode_step_latency.png
```
