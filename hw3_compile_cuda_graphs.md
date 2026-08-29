# HW3 — `torch.compile` & CUDA Graphs

**Lecture mapping:** L2 §03 (compilation, CUDA graphs) · L1 §07 (profiling)

At batch size 1, a decode step is hundreds of tiny kernels, each finishing faster
than the CPU can launch the next — so generation speed is set by launch overhead,
not FLOPs or bandwidth. You'll measure that overhead and remove it two ways:

1. **Part 1** — measure per-launch overhead and find the tensor size below which a
   kernel is effectively free (launch-bound).
2. **Part 2** — `torch.compile`: what graph breaks are, how to find them with
   `torch._dynamo.explain`, and how to rewrite a decode step to compile as one
   graph (`fullgraph=True`).
3. **Part 3** — capture a decode step into a CUDA graph by hand with
   `torch.cuda.CUDAGraph` (static buffers, side-stream warmup, replay) — what
   `torch.compile(mode="reduce-overhead")` automates.

Parts 1–2 run anywhere; Part 3 and the final benchmark need a GPU.

## What you implement

| Part | Function |
|------|----------|
| 1 | `sweep_elementwise_times` + `estimate_launch_overhead_us` |
| 2 | `fixed_decode_step` — same math as the broken step, zero graph breaks |
| 3 | `make_graphed_callable` — manual CUDA-graph capture + replay |
| 4 | Writeup Q1–Q3 |

## Cell types

- **DO NOT EDIT** — fixed harness.
- **YOUR IMPLEMENTATION** — replace `raise NotImplementedError`.
- **SELF-CHECK** — asserts that must pass.
- **WRITEUP** — answer in the markdown cell.

Run top-to-bottom on the GPU. Submit the executed notebook plus the files under
`results/`.

## Setup (DO NOT EDIT)


```python
# DO NOT EDIT — device + a known-good timer.
import contextlib, io, math, os, time

import torch
from torch import _dynamo as dynamo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join("results", "hw3")
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.set_grad_enabled(False)   # inference-only homework

print(f"torch {torch.__version__}  device={DEVICE}"
      + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))


def bench(fn, n_warmup: int = 10, n_iters: int = 100) -> float:
    """Average seconds per call — CUDA-event timing on GPU, perf_counter on CPU.
    (This is the timer you built yourself in HW1.)"""
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
        return start.elapsed_time(end) / 1e3 / n_iters
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters
```

## Part 1 — cost of one kernel launch

A CUDA kernel launch costs the CPU a few microseconds (driver call, queueing)
regardless of how much work the kernel does. For a tiny tensor the GPU finishes
almost instantly, so the wall time is nearly pure launch overhead; only past some
size does the actual work dominate. Sweep one elementwise op across sizes and
find that knee.


```python
# TODO-CELL: sweep_elementwise_times
# YOUR IMPLEMENTATION
def sweep_elementwise_times(sizes: list) -> list:
    """Time ONE out-of-place elementwise op (x + 1.0) over a 1-D fp32 tensor of
    n elements on DEVICE, for each n in sizes. Same op for every size."""
    times = []
    for n in sizes:
        x = torch.ones(n, device=DEVICE, dtype=torch.float32)
        times.append(bench(lambda x=x: x + 1.0))
    return times


def estimate_launch_overhead_us(sizes: list, times: list) -> float:
    """Average seconds-per-call over the clearly launch-bound sizes (n <= 2**12)
    and convert to microseconds."""
    vals = [t for n, t in zip(sizes, times) if n <= 2 ** 12]
    return (sum(vals) / len(vals)) * 1e6

```


```python
# DO NOT EDIT — run the sweep + plot.
SIZES = [2 ** k for k in range(0, 25, 2)]
times = sweep_elementwise_times(SIZES)
assert len(times) == len(SIZES), "one measurement per size"
overhead_us = estimate_launch_overhead_us(SIZES, times)

for n, t in zip(SIZES, times):
    print(f"n = {n:>10,}   {t * 1e6:8.2f} us/call")
print(f"\nestimated per-launch overhead = {overhead_us:.2f} us")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(SIZES, [t * 1e6 for t in times], "o-")
ax.axhline(overhead_us, ls=":", c="gray",
           label=f"launch overhead = {overhead_us:.1f} us")
ax.set_xlabel("elements per op")
ax.set_ylabel("time per call (us)")
ax.set_title(f"Elementwise op latency vs size — {DEVICE}")
ax.legend()
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "launch_overhead.png"), dpi=130)
plt.show()
```

## Part 2 — `torch.compile` and graph breaks

`torch.compile` traces Python into a graph (TorchDynamo) and compiles it into
fused kernels (TorchInductor). The trace breaks whenever the code needs a real
Python value from tensor data: `.item()`, `print(tensor)`, an `if` on a tensor.
Each break splits the function into separate compiled chunks with eager glue
between them (no fusion across the break), and `.item()` also forces the CPU to
wait for the GPU.

`broken_decode_step` below uses three such constructs. Run the `explain` cell to
see the breaks (it may report fewer than three — once dynamo falls back to Python
it can swallow several in one gap). Then write `fixed_decode_step`: the same math
(minus the print), compilable as one graph.


```python
# DO NOT EDIT — the offender.
def broken_decode_step(h: torch.Tensor, state: torch.Tensor):
    """One step of a toy recurrent decode update. Correct, but hard to compile."""
    # 1) rms-normalize
    h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    # 2) rescale if the max |value| is large — via a host round-trip
    scale = h.abs().max().item()          # GPU -> CPU sync + graph break
    if scale > 3.0:                       # Python branch on tensor data
        h = h / scale
    # 3) progress logging on the hot path
    print(f"max|h| = {scale:.3f}")        # graph break
    # 4) EMA state update
    state = 0.9 * state + 0.1 * h
    return h, state
```


```python
# DO NOT EDIT — dynamo's diagnosis of the offender.
dynamo.reset()
_h = torch.randn(2, 64, device=DEVICE)
_s = torch.zeros(2, 64, device=DEVICE)
with contextlib.redirect_stdout(io.StringIO()):  # mute the print inside
    explanation = dynamo.explain(broken_decode_step)(_h, _s)
print(f"graphs produced : {explanation.graph_count}")
print(f"graph breaks    : {explanation.graph_break_count}")
for i, reason in enumerate(explanation.break_reasons):
    print(f"  break {i}: {str(reason.reason).splitlines()[0]}")
```


```python
# TODO-CELL: fixed_decode_step
# YOUR IMPLEMENTATION
def fixed_decode_step(h: torch.Tensor, state: torch.Tensor):
    """Same math as broken_decode_step (minus the print), with ZERO graph breaks."""
    # 1) rms-normalize
    h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    # 2) rescale if the max |value| is large -- keep scale as a 0-dim TENSOR
    #    (no .item()) and branch with torch.where (stays on the GPU).
    scale = h.abs().max()
    h = torch.where(scale > 3.0, h / scale, h)
    # 3) (print dropped -- no logging on the hot path)
    # 4) EMA state update
    state = 0.9 * state + 0.1 * h
    return h, state

```


```python
# SELF-CHECK — DO NOT EDIT.
torch.manual_seed(0)

_big = torch.zeros(2, 64, device=DEVICE); _big[0, 0] = 100.0  # forces the rescale branch
_flat = torch.ones(2, 64, device=DEVICE)                      # avoids it
_cases = [_big, _flat] + [torch.randn(2, 64, device=DEVICE) * 3 for _ in range(4)]

for _i, _hc in enumerate(_cases):
    _sc = torch.randn(2, 64, device=DEVICE)
    with contextlib.redirect_stdout(io.StringIO()):  # mute broken's print
        _h_ref, _s_ref = broken_decode_step(_hc.clone(), _sc.clone())
    _h_new, _s_new = fixed_decode_step(_hc.clone(), _sc.clone())
    assert torch.allclose(_h_ref, _h_new, atol=1e-6), f"h mismatch on case {_i}"
    assert torch.allclose(_s_ref, _s_new, atol=1e-6), f"state mismatch on case {_i}"
print("same math as the broken step       PASS")

# backend="eager" skips kernel codegen — this tests ONLY the dynamo trace,
# so it passes/fails identically on CPU and GPU.
dynamo.reset()
_compiled = torch.compile(fixed_decode_step, fullgraph=True, backend="eager")
_compiled(torch.randn(2, 64, device=DEVICE), torch.zeros(2, 64, device=DEVICE))
print("compiles with fullgraph=True       PASS")

dynamo.reset()
_ex = dynamo.explain(fixed_decode_step)(torch.randn(2, 64, device=DEVICE),
                                        torch.zeros(2, 64, device=DEVICE))
assert _ex.graph_break_count == 0, f"{_ex.graph_break_count} graph break(s) remain"
print("0 graph breaks                     PASS")
print()
print("All checks passed")
```


```python
# DO NOT EDIT — what the fix buys: broken-eager vs fixed-compiled.
_h = torch.randn(2, 64, device=DEVICE)
_s = torch.zeros(2, 64, device=DEVICE)

with contextlib.redirect_stdout(io.StringIO()):   # mute 100+ prints
    t_broken = bench(lambda: broken_decode_step(_h, _s), n_warmup=3, n_iters=50)

dynamo.reset()
_compiled_fixed = torch.compile(fixed_decode_step)
try:
    t_fixed = bench(lambda: _compiled_fixed(_h, _s), n_warmup=10, n_iters=50)
    tag = "compiled, one graph"
except Exception as e:
    print(f"inductor backend unavailable here ({type(e).__name__}) — timing eager fixed instead.")
    t_fixed = bench(lambda: fixed_decode_step(_h, _s), n_warmup=3, n_iters=50)
    tag = "eager (compile unavailable)"

print(f"broken (eager, sync + print): {t_broken*1e6:8.1f} us/step")
print(f"fixed  ({tag}): {t_fixed*1e6:8.1f} us/step   ({t_broken/t_fixed:.1f}x)")
if DEVICE == "cpu":
    print()
    print("Note: on CPU a 'sync' costs nothing (CPU execution is synchronous"
          " already), so the gap here is small or even negative. The GPU run"
          " is the one that counts.")
```

## Part 3 — CUDA graphs by hand

A CUDA graph records the exact sequence of kernels (and their memory addresses)
once, then replays the whole sequence with a single launch — turning N launches
into 1, which is what a launch-bound decode loop needs.

The cost is rigidity: replay uses the same memory and same shapes every time. So:
- inputs must live in a static buffer you `copy_` new data into;
- outputs come back in a static buffer that the next replay overwrites;
- nothing inside may sync with the host, allocate new memory, or change shape.

`torch.compile(mode="reduce-overhead")` does this for you; here you do it by hand.
The step below is a 4-block MLP at batch 1 — dozens of small kernels.


```python
# DO NOT EDIT — the toy decode step.
class TinyDecoder(torch.nn.Module):
    def __init__(self, d: int = 256, hidden: int = 1024, blocks: int = 4):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.LayerNorm(d),
                torch.nn.Linear(d, hidden),
                torch.nn.GELU(),
                torch.nn.Linear(hidden, d),
            ) for _ in range(blocks))

    def forward(self, x):
        for blk in self.blocks:
            x = x + blk(x)
        return x


torch.manual_seed(0)
decoder = TinyDecoder().to(DEVICE).eval()
for p in decoder.parameters():
    p.requires_grad_(False)
example = torch.randn(1, 256, device=DEVICE)
print(decoder(example).shape)
```


```python
# TODO-CELL: make_graphed_callable
# YOUR IMPLEMENTATION
def make_graphed_callable(fn, example_input: torch.Tensor):
    """Capture `fn(example_input)` into a CUDA graph; return `graphed(x)` that
    replays it for new inputs of the same shape/dtype."""
    static_in = example_input.clone()

    # Warm up on a side stream (capture fails on a stream with pending work).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(s)

    # Capture the kernel sequence once.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = fn(static_in)

    def graphed(x: torch.Tensor) -> torch.Tensor:
        static_in.copy_(x)          # feed new data into the captured input buffer
        g.replay()                  # one launch replays the whole sequence
        return static_out.clone()   # next replay overwrites static_out

    return graphed

```


```python
# SELF-CHECK — DO NOT EDIT. (GPU only — SKIPs on CPU.)
if DEVICE == "cuda":
    graphed = make_graphed_callable(decoder, example)
    for _i in range(3):
        _xin = torch.randn(1, 256, device=DEVICE)
        _ref = decoder(_xin)
        _out = graphed(_xin)
        assert torch.allclose(_ref, _out, atol=1e-5), f"mismatch vs eager on input {_i}"
    print("matches eager on fresh inputs      PASS")

    _a = graphed(torch.randn(1, 256, device=DEVICE))
    _b = graphed(torch.randn(1, 256, device=DEVICE))
    assert not torch.allclose(_a, _b), \
        "two different inputs gave the same output — did you copy_ into the static buffer and clone the output?"
    print("static buffers handled correctly   PASS")
    print()
    print("All checks passed")
else:
    print("SKIP — CUDA graphs need a GPU. Run this notebook on your Nebius instance.")
```

### Final benchmark — 256-step decode, four ways

Each variant generates tokens sequentially (the output of step *t* is the input
of *t+1*). If `reduce-overhead` errors here about overwritten outputs, that's the
static-buffer constraint — explain it in Q3 rather than working around it.


```python
# DO NOT EDIT — the final benchmark. (GPU only.)
if DEVICE == "cuda":
    N_STEPS = 256

    def run_decode(step_fn, x0, n_steps=N_STEPS):
        x = x0
        for _ in range(n_steps):
            x = step_fn(x)
        return x

    variants = {"eager": decoder,
                "manual CUDA graph": make_graphed_callable(decoder, example)}
    try:
        dynamo.reset()
        variants["torch.compile (default)"] = torch.compile(decoder)
        variants["torch.compile (reduce-overhead)"] = torch.compile(
            decoder, mode="reduce-overhead")
    except Exception as e:
        print(f"torch.compile unavailable: {type(e).__name__}: {e}")

    times_step = {}
    for name, fn in variants.items():
        try:
            t = bench(lambda: run_decode(fn, example), n_warmup=3, n_iters=10)
            times_step[name] = t / N_STEPS
        except Exception as e:
            print(f"{name}: FAILED ({type(e).__name__}: {e}) "
                  f"— see Q3 on CUDA-graph constraints")

    base = times_step["eager"]
    print()
    for name, t in times_step.items():
        print(f"{name:34s} {t*1e6:8.1f} us/step   {base/t:5.2f}x vs eager")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(times_step)
    ax.bar(names, [times_step[n] * 1e6 for n in names], color="tab:blue")
    ax.set_ylabel("us per decode step")
    ax.set_title(f"{N_STEPS}-step decode at batch 1 — per-step latency")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "decode_step_latency.png"), dpi=130)
    plt.show()
else:
    print("SKIP — the benchmark needs a GPU.")
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

Related: if input shapes change every call (e.g., a KV-cache decode loop without `StaticCache`, where sequence length grows from 65 to 192), dynamo recompiles for each new shape, spending more time recompiling than it saves. Detection: the profiler trace shows repeated "recompile" events; the profiler table shows `torch._dynamo.compile` entries accumulating time proportional to the number of distinct shapes. Seen in HW2: `compile(dynamic=True)` over a growing KV cache reached only 2.54× vs 4.62× with `StaticCache` (fixed shapes, one capture).
