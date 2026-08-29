# Status — what's done so far

## Code: all 9 functions implemented

All `# TODO-CELL` blocks across the three notebooks have been filled in.
`.md` mirrors (`hwN_*.md`) are kept in sync with the `.ipynb` files.

| Notebook | Function | Approach |
|---|---|---|
| HW1 | `lowest_ai_fn` | `lambda: x * 2.0` — 1 read + 1 write, ~1 FLOP/elem |
| HW1 | `make_compute_fn(x, k)` | chains `k` steps of `y = y * 1.0001 + 0.5` (2 FLOP/elem/step, matches `flops=2*k*N`), no mutation of `x` |
| HW1 | `benchmark_fn` | warmup loop, then `torch.cuda.Event` timing + sync (CPU fallback: `perf_counter`) |
| HW1 | `compute_metrics` | returns `{"ai", "achieved_tflops", "achieved_gbps"}` from flops/bytes/seconds |
| HW2 | `profile` | `torch.profiler.profile` (CPU+CUDA activities), prints `key_averages().table(...)`, exports chrome trace |
| HW2 | `optimized_loop` | KV-cache prefill + single-token decode steps (O(n) instead of baseline's O(n²)); no `.item()` on the hot path — token-exact match with `baseline_loop` in fp32 |
| HW2 | `generate_optimized` | bf16 + KV cache, timed via `time_loop` |
| HW3 | `sweep_elementwise_times` + `estimate_launch_overhead_us` | times `x + 1.0` across sizes; overhead = avg time for `n <= 2**12`, in µs |
| HW3 | `fixed_decode_step` | same math as `broken_decode_step`, `scale` kept as 0-dim tensor, `torch.where` replaces the Python `if`, `print` dropped — 0 graph breaks under `fullgraph=True` |
| HW3 | `make_graphed_callable` | side-stream warmup → `torch.cuda.graph()` capture → `copy_`/`replay()`/`clone()` replay wrapper |

## Local validation (Mac, CPU-only, via isolated `uv` venv `.venv/`)

Installed CPU-only `torch`, `transformers`, `matplotlib`, `jupyter`/`nbconvert` into
`.venv/` (created with `uv venv` + `uv pip install`). Ran each notebook's
CPU-reachable cells as extracted scripts (with `KMP_DUPLICATE_LIB_OK=TRUE
MPLBACKEND=Agg` to work around a macOS-only OpenMP/GUI quirk — see note below).

- **HW1** — ran cells 0–6 (setup → self-check). All asserts **PASS**:
  `compute_metrics`, `lowest_ai_fn`, `make_compute_fn`, `benchmark_fn`.
  (Cell 7 — the GPU sweep/plot that calls `torch.compile` — was *not* run
  locally; see note below.)
- **HW2** — ran the full notebook (CPU auto-trims to 32 tokens). Results:
  - `optimized_loop matches baseline   PASS` (token-exact fp32 KV-cache decode)
  - `profile() writes a Chrome trace   PASS`
  - End-to-end on CPU: baseline 194.2 ms vs optimized 44.9 ms → **4.33x**
    ("excellent" tier) — promising, but the *graded* number must come from a GPU run.
- **HW3** — ran all cells except the one `torch.compile` "broken vs fixed" timing
  cell (DO NOT EDIT, see note below). Results:
  - Launch-overhead sweep produced a sensible curve (knee around 64K–1M
    elements, ~1.36 µs estimated overhead).
  - `fixed_decode_step`: **same math as broken step**, **0 graph breaks**,
    compiles with `fullgraph=True` — all **PASS**.
  - Part 3 self-check and final benchmark correctly printed
    `SKIP — ... needs a GPU` (expected on CPU).

**Note — macOS-only `torch.compile` deadlock:** On this Mac (M4 Pro), two cells
that call `torch.compile` with the default *inductor* backend (HW1's sweep cell,
HW3's "broken vs fixed compiled" cell) **hang/deadlock** in `libomp` — a known
macOS issue caused by two conflicting OpenMP runtimes (PyTorch's bundled libomp
vs. anaconda/Homebrew's) inside inductor-generated C++ kernels. This is an
environment artifact, not a bug in the implemented code — on a CUDA GPU,
inductor uses Triton instead of the OpenMP C++ backend, so it will not occur.
These two cells were skipped locally and must be run for real on the GPU.

## Cleanup done

- Removed smoke-test artifacts (`results/hw2/trace_*.json`,
  `results/hw3/launch_overhead.png`) — these were CPU-trimmed/dummy and would
  be overwritten by the real GPU run anyway.
- Notebooks have **no stray outputs** — `execution_count: null`, `outputs: []`
  for every code cell. Clean slate for a top-to-bottom GPU run.
- `.venv/` (uv-managed, CPU-only) remains in the project dir for future local
  debugging of logic (not for grading).

## Not yet done

- The actual GPU execution (Colab or Nebius) — see `gpu_run.md`.
- The four writeup sections (HW1 Q1–Q2, HW2 Q1–Q4, HW3 Q1–Q3) — answer outlines
  are in `asgmnt_plan.md`, but should be written from the real GPU numbers/plots.
- Submission (executed notebooks + `results/` folders).
