# HW1 — Roofline & Arithmetic Intensity

**Lecture mapping:** L1 §06 (Roofline) · L1 §03 (GPU architecture)

Measure real GPU kernels and plot them on the roofline for your card.

```
achievable_perf = min( peak_compute,  peak_bandwidth × arithmetic_intensity )
arithmetic_intensity (FLOP/byte) = FLOPs / bytes moved to-from HBM
```

The ridge point is at `peak_compute / peak_bandwidth` (≈ 295 FLOP/byte on an
H100). Left of it: memory-bound. Right of it: compute-bound.

## What you implement

| Part | Function |
|------|----------|
| 1a | `lowest_ai_fn` — lowest-arithmetic-intensity elementwise op |
| 1b | `make_compute_fn` — tunable-K kernel, eager and compiled |
| 2  | `benchmark_fn` — timing with CUDA events + warmup |
| 3  | `compute_metrics` — measurements → (AI, TFLOP/s, GB/s) |
| 4  | Writeup Q1–Q2 |

## Cell types

- **DO NOT EDIT** — fixed harness.
- **YOUR IMPLEMENTATION** — replace `raise NotImplementedError`.
- **SELF-CHECK** — asserts that must pass.
- **WRITEUP** — answer in the markdown cell.

Run top-to-bottom on the GPU (H100 / L40S). The self-check also runs on CPU.
Submit the executed notebook plus the files under `results/`.

## Setup — hardware specs & helpers (DO NOT EDIT)


```python
# DO NOT EDIT — hardware specs, dtype helpers.
from dataclasses import dataclass
from typing import Callable
import json, math, os, time

import torch

RESULTS_DIR = os.path.join("results", "hw1")
os.makedirs(RESULTS_DIR, exist_ok=True)


@dataclass(frozen=True)
class GpuSpec:
    name: str
    peak_flops: float   # peak tensor FLOP/s for the benchmark dtype (BF16)
    peak_bw: float      # peak HBM bandwidth, bytes/s
    vram_gb: float

    @property
    def ridge_ai(self) -> float:
        """Arithmetic intensity (FLOP/byte) at the roofline ridge point."""
        return self.peak_flops / self.peak_bw


# Vendor spec sheets. BF16 tensor-core peak (dense, no sparsity) and HBM BW.
H100_SXM = GpuSpec("H100 SXM", peak_flops=989e12, peak_bw=3.35e12, vram_gb=80)
L40S = GpuSpec("L40S", peak_flops=362e12, peak_bw=0.864e12, vram_gb=48)
# CPU fallback so the notebook runs for a smoke test off-GPU. The numbers are
# placeholders — only GPU runs produce meaningful results.
CPU_FALLBACK = GpuSpec("CPU (fallback)", peak_flops=1e12, peak_bw=50e9, vram_gb=8)


def detect_hardware() -> GpuSpec:
    if not torch.cuda.is_available():
        print("CUDA not available — using CPU fallback spec. "
              "Results are NOT meaningful; run on H100/L40S.")
        return CPU_FALLBACK
    name = torch.cuda.get_device_name(0)
    if "H100" in name:
        return H100_SXM
    if "L40" in name:
        return L40S
    print(f"Unrecognized GPU '{name}'. Defaulting to H100 ceilings; "
          f"ask the instructor to add your card's spec.")
    return H100_SXM


def benchmark_dtype() -> torch.dtype:
    """BF16 on GPU (tensor cores, matches the spec peak); float32 on CPU."""
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


SPEC = detect_hardware()
DTYPE = benchmark_dtype()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE} dtype={DTYPE} spec={SPEC.name} "
      f"peak={SPEC.peak_flops/1e12:.0f} TFLOP/s bw={SPEC.peak_bw/1e12:.2f} TB/s "
      f"ridge≈{SPEC.ridge_ai:.0f} FLOP/byte")
```


```python
# DO NOT EDIT — roofline plotting.
def plot_roofline(points: list, spec: GpuSpec, out_path: str) -> None:
    """Render measured (AI, achieved TFLOP/s) points under the roofline.

    points: list of {"label", "ai", "achieved_tflops", "series"}.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))

    # Roofline: memory ceiling (slope = BW) then flat compute ceiling.
    ai_grid = np.logspace(-2, 4, 400)
    peak_tflops = spec.peak_flops / 1e12
    bw_tflops_per_ai = spec.peak_bw / 1e12  # (TFLOP/s) per (FLOP/byte)
    roof = np.minimum(peak_tflops, bw_tflops_per_ai * ai_grid)
    ax.plot(ai_grid, roof, "k-", lw=2, label="roofline")
    ax.axhline(peak_tflops, ls="--", c="gray", lw=1)
    ax.axvline(spec.ridge_ai, ls=":", c="gray", lw=1,
               label=f"ridge ≈ {spec.ridge_ai:.0f} FLOP/byte")

    series = {}
    for p in points:
        series.setdefault(p.get("series", "ops"), []).append(p)
    for name, pts in series.items():
        pts = sorted(pts, key=lambda d: d["ai"])
        xs = [d["ai"] for d in pts]
        ys = [d["achieved_tflops"] for d in pts]
        ax.plot(xs, ys, "o-", ms=7, label=name)
        for d in pts:
            ax.annotate(d["label"], (d["ai"], d["achieved_tflops"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Achieved performance (TFLOP/s)")
    ax.set_title(f"Roofline — {spec.name}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.show()
    print(f"wrote {out_path}")
```

## Part 1a — lowest-AI op

Return a no-argument function that does one elementwise pass over `x` and returns
a same-size tensor. Aim for the lowest arithmetic intensity: a few FLOPs per
element, one read and one write. This should land on the memory-bound (left)
side of the roofline.


```python
# TODO-CELL: lowest_ai_fn
# YOUR IMPLEMENTATION
def lowest_ai_fn(x: torch.Tensor) -> Callable[[], torch.Tensor]:
    """Return a no-argument function doing one cheap elementwise pass over x.

    One read of x, one write of y, ~1 FLOP/element -> lowest arithmetic
    intensity, lands on the memory-bound (left) side of the roofline.
    """
    return lambda: x * 2.0

```

## Part 1b — tunable-intensity kernel

Return a no-argument function that applies `k` chained elementwise ops to `x`,
each feeding into the next. Ideal arithmetic intensity grows with `k`: you load
`x` once and do `k×` the work. In eager mode each step is a separate kernel with
its own HBM round-trip, so achieved intensity stays low. The run cell also wraps
your function in `torch.compile`, which fuses the steps and moves the point
toward the compute roof.

Chain real data dependencies through `x` so the steps can't be constant-folded.


```python
# TODO-CELL: make_compute_fn
# YOUR IMPLEMENTATION
def make_compute_fn(x: torch.Tensor, k: int) -> Callable[[], torch.Tensor]:
    """Return a no-argument function applying k chained elementwise ops to x.
    Must not mutate x."""
    def run() -> torch.Tensor:
        y = x  # never mutated in place; each op below returns a fresh tensor
        for _ in range(k):
            y = y * 1.0001 + 0.5  # mul + add = 2 FLOP/element/step (matches flops=2*k*N)
        return y
    return run

```

## Part 2 — timing with CUDA events

GPU launches are asynchronous, so `time.time()` measures how fast Python queued
the kernel, not how long the GPU ran it. Time with
`torch.cuda.Event(enable_timing=True)` and synchronize before reading.


```python
# TODO-CELL: benchmark_fn
# YOUR IMPLEMENTATION
def benchmark_fn(fn: Callable[[], torch.Tensor],
                 n_warmup: int = 10, n_iters: int = 50) -> float:
    """Time ``fn`` and return the AVERAGE wall time per call, in SECONDS."""
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return (start.elapsed_time(end) / 1e3) / n_iters  # ms -> s, per call
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters

```

## Part 3 — measurements → roofline coordinates


```python
# TODO-CELL: compute_metrics
# YOUR IMPLEMENTATION
def compute_metrics(flops: float, bytes_moved: float, seconds: float) -> dict:
    """Convert a measurement into roofline coordinates."""
    return {
        "ai": flops / bytes_moved,
        "achieved_tflops": flops / seconds / 1e12,
        "achieved_gbps": bytes_moved / seconds / 1e9,
    }

```

## Self-check

Fast asserts (run anywhere). Passing doesn't mean the plot and writeup are done;
failing means something is wrong.


```python
# SELF-CHECK — DO NOT EDIT.
torch.manual_seed(0)
_x = torch.randn(4096).abs() + 0.1

m = compute_metrics(flops=2e12, bytes_moved=1e9, seconds=1e-3)
assert set(m) == {"ai", "achieved_tflops", "achieved_gbps"}, set(m)
assert math.isclose(m["ai"], 2000, rel_tol=1e-6), m["ai"]
assert math.isclose(m["achieved_tflops"], 2000, rel_tol=1e-6), m["achieved_tflops"]
assert math.isclose(m["achieved_gbps"], 1000, rel_tol=1e-6), m["achieved_gbps"]
print("compute_metrics   PASS")

_y = lowest_ai_fn(_x)()
assert torch.is_tensor(_y) and _y.numel() == _x.numel(), "must return a same-size tensor"
print("lowest_ai_fn      PASS")

_x0 = _x.clone()
_g1 = make_compute_fn(_x, 1)()
_g8 = make_compute_fn(_x, 8)()
assert torch.allclose(_x, _x0), "make_compute_fn must not mutate its input"
assert torch.is_tensor(_g8), "must return a tensor"
assert not torch.allclose(_g1, _g8), "k must actually change the result (real data dependency)"
print("make_compute_fn   PASS")

_t_light = benchmark_fn(lambda: _x * 2.0, n_warmup=2, n_iters=5)
_t_heavy = benchmark_fn(
    lambda: _x.unsqueeze(0).repeat(256, 1) @ _x.unsqueeze(1).repeat(1, 256),
    n_warmup=2, n_iters=5)
assert isinstance(_t_light, float) and _t_light > 0, "must return positive seconds as float"
assert _t_heavy >= _t_light * 0.5, "a much heavier op should not be much faster"
print("benchmark_fn      PASS")
print()
print("All checks passed")
```

## Run on the GPU

Benchmarks ops from low to high arithmetic intensity with your `benchmark_fn`,
converts them with your `compute_metrics`, and plots the roofline.

A correct plot:
- all points sit below the roofline;
- `elementwise` and small-`K` `compute-K` sit low on the left (memory-bound);
- `matmul-N` climbs up and to the right toward the compute ceiling as `N` grows;
- compiled `compute-K` sits above eager at the same intensity (what compilation buys).


```python
# DO NOT EDIT — the sweep + plot. (Trimmed automatically on CPU.)
ELEM_N = 1 << 24 if torch.cuda.is_available() else 1 << 18      # vector length
COMPUTE_KS = [1, 4, 16, 64]                                     # ops per element
MATMUL_NS = [1024, 2048, 4096] if torch.cuda.is_available() else [128, 256]


def _try_compile(fn):
    try:
        c = torch.compile(fn)
        c()  # force a trace now so failures surface here
        return c
    except Exception as e:
        print(f"torch.compile unavailable ({type(e).__name__}); "
              f"using eager for the compiled series.")
        return fn


dbytes = dtype_bytes(DTYPE)
points = []

# --- Part 1a: lowest arithmetic intensity (memory-bound) ----------------
x = torch.randn(ELEM_N, device=DEVICE, dtype=DTYPE).abs() + 0.1
fn = lowest_ai_fn(x)
t = benchmark_fn(fn)
# Convention for this op: ~1 FLOP/element, read x + write y.
m = compute_metrics(flops=ELEM_N, bytes_moved=2 * ELEM_N * dbytes, seconds=t)
points.append({"label": "elementwise", "series": "memory-bound",
               "ai": m["ai"], "achieved_tflops": m["achieved_tflops"]})
print(f"elementwise: {t*1e6:8.1f} us  AI={m['ai']:.3g}  {m['achieved_gbps']:.0f} GB/s")

# --- Part 1b: compute-K, eager vs compiled ------------------------------
for tag, compiler in (("eager", lambda f: f), ("compiled", _try_compile)):
    for k in COMPUTE_KS:
        base = make_compute_fn(x, k)
        fn = compiler(base)
        t = benchmark_fn(fn)
        flops = 2 * k * ELEM_N                 # mul+add per step
        ideal_bytes = 2 * ELEM_N * dbytes      # load x once, store once
        m = compute_metrics(flops=flops, bytes_moved=ideal_bytes, seconds=t)
        points.append({"label": f"K={k}", "series": f"compute-K ({tag})",
                       "ai": m["ai"], "achieved_tflops": m["achieved_tflops"]})
    print(f"compute-K [{tag}] done")

# --- matmul sweep: anchors the compute-bound side ------------------------
for n in MATMUL_NS:
    a = torch.randn(n, n, device=DEVICE, dtype=DTYPE)
    b = torch.randn(n, n, device=DEVICE, dtype=DTYPE)
    fn = (lambda a=a, b=b: a @ b)
    t = benchmark_fn(fn)
    flops = 2 * n ** 3
    bytes_moved = 3 * n * n * dbytes           # load a, load b, store c
    m = compute_metrics(flops=flops, bytes_moved=bytes_moved, seconds=t)
    points.append({"label": f"N={n}", "series": "matmul-N",
                   "ai": m["ai"], "achieved_tflops": m["achieved_tflops"]})
    print(f"matmul N={n}: {t*1e3:7.3f} ms  AI={m['ai']:.3g}  "
          f"{m['achieved_tflops']:.0f} TFLOP/s")

# --- emit artifacts -------------------------------------------------------
out_json = os.path.join(RESULTS_DIR, "roofline_data.json")
with open(out_json, "w") as f:
    json.dump({"spec": SPEC.name, "dtype": str(DTYPE),
               "peak_tflops": SPEC.peak_flops / 1e12,
               "peak_bw_tbps": SPEC.peak_bw / 1e12,
               "points": points}, f, indent=2)
print(f"wrote {out_json}")
plot_roofline(points, SPEC, os.path.join(RESULTS_DIR, "roofline.png"))
```

---
## WRITEUP

### Q1

Look at your roofline. Which of your points are memory-bound and which are
compute-bound? How can you tell **from the plot alone**?

**Your answer:**

A point is **memory-bound** if it falls on the rising BW slope (achieved TFLOP/s ≈ AI × peak_BW) and **compute-bound** if it hugs the flat compute ceiling. A point that *appears* to float far **below** both ceilings is usually being plotted at an **idealized arithmetic intensity the kernel never actually achieves** — at its *true* AI (the bytes the hardware really moves) it falls back onto one of the ceilings.

**`elementwise` (AI = 0.25):** Lands on the left side of the roofline, on the BW slope. At 0.65 TFLOP/s vs the BW ceiling of 0.25 × 3350 = 0.84 TFLOP/s (78% utilization), it is **memory-bound**.

**`compute-K` eager (K = 1–64):** All four sit flat at ≈ 0.65 TFLOP/s and *appear* to float far below both ceilings — but that is an artifact of plotting them at the **ideal load-once AI** (which grows with K). In eager mode the chain is never fused: each of the 2K ops (a mul and an add per step) is its own kernel that re-reads the whole tensor from HBM and writes it back (~67 MB round-trip, ≈ 25.8 µs each — exactly the `elementwise` cost). So the **real** arithmetic intensity stays pinned at ≈ 0.25 FLOP/byte regardless of K (the bytes moved scale with K just as the FLOPs do), and 0.65 TFLOP/s is the same ~77% of the 0.84 TFLOP/s BW ceiling as `elementwise`. These points are **memory-bound** — at their true AI they sit right on the BW slope, exactly as the assignment notes: *“each step is a separate kernel with its own HBM round-trip, so achieved intensity stays low.”* Launch overhead (≈ 6 µs/kernel, measured in HW3) is only ~23% of each 25.8 µs kernel here, not the dominant cost on these 16 M-element ops.

**`compute-K` compiled (K = 1–64):** Compilation **fuses the whole chain into one kernel**, so x is loaded from HBM once and the K ops run in registers (one round-trip instead of K). That raises the *real* arithmetic intensity, so achieved TFLOP/s now climbs with K instead of staying flat. At K = 64 we reach 50.6 TFLOP/s — 47% of the 32 × 3.35 = 107 TFLOP/s BW ceiling — **climbing the BW slope toward memory-bound**: fusion converted a per-kernel-round-trip-bound workload into a bandwidth-bound one. This is the concrete win compilation buys, visible as the compiled series sitting above eager at the same plotted AI.

**`matmul-N` (N = 1024–4096):** AI = 341–1365, well to the right of the ridge point (≈ 295 FLOP/byte). These points climb toward the flat **compute ceiling** (989 TFLOP/s). N = 1024 reaches only 20% utilization (too small to saturate the tensor cores), but N = 4096 reaches 74% — clearly **compute-bound**, sitting near the flat top.

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
