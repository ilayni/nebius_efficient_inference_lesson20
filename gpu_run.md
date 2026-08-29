# Next steps — running on a GPU

All code is implemented and CPU-validated (see `asgmnt_status.md`). What remains
needs an NVIDIA GPU. Pick **Nebius L40S/H100** (preferred — HW1's hardcoded specs
match) or **Colab** (works for HW2/HW3 ratios; HW1 roofline only calibrated
correctly if your runtime reports `"H100"` or `"L40"` in the device name — see
`asgmnt_plan.md` for the Colab caveat).

## 0. Get the files onto the GPU box

Upload (or `scp`/clone) these three notebooks:
- `hw1_roofline.ipynb`
- `hw2_decode_optimization.ipynb`
- `hw3_compile_cuda_graphs.ipynb`

(`.md` files and `asgmnt_*.md` are just your notes — don't need to go to the GPU.)

## 1. Environment setup (once per GPU box)

```bash
pip install torch transformers matplotlib numpy
```

Sanity check before running anything:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
- If this prints `True` and a name containing `H100` or `L40` → HW1's roofline
  ceilings will be correct automatically.
- Any other name (T4, A100, L4, ...) → HW1 still runs, but the plotted roofline
  ceiling defaults to H100 numbers (a known limitation of the DO-NOT-EDIT setup
  cell). Note this when answering HW1 Q1/Q2 if it applies to you.

## 2. HW1 — `hw1_roofline.ipynb`

1. Run **all cells top to bottom**.
2. Watch for:
   - The setup cell prints `device=cuda dtype=torch.bfloat16 spec=<your GPU> ...`
   - The sweep cell (the one with `torch.compile` / `_try_compile`) — on GPU this
     uses Triton, so it should NOT hang (unlike the Mac). If it does throw, the
     `_try_compile` fallback catches it and prints a message — that's fine.
3. **Check the output plot** (`results/hw1/roofline.png`):
   - all points sit below the roofline
   - `elementwise` and small-K `compute-K` are low/left (memory-bound)
   - `matmul-N` climbs up-right toward the compute ceiling as N grows
   - compiled `compute-K` sits above eager `compute-K` at the same AI
4. **Write HW1 Q1 & Q2** in the WRITEUP markdown cells, referencing your actual
   plot/numbers. Outline answers are in `asgmnt_plan.md`.
5. Confirm `results/hw1/roofline_data.json` and `results/hw1/roofline.png` exist.

## 3. HW2 — `hw2_decode_optimization.ipynb`

1. Run **all cells top to bottom**.
2. The harness will print:
   ```
   baseline V0: ... ms/generation
   optimized_loop correctness (fp32): PASS
   optimized:   ... ms/generation
   speedup: X.XXx  -> <tier>
   ```
   - **Correctness must say PASS.** If it doesn't, something about the KV-cache
     loop diverges from the baseline on this GPU/dtype — stop and debug before
     trusting the speedup number.
   - Target tiers: ≥4.0x excellent, ≥3.0x good, ≥1.5x some speedup. On CPU this
     already hit 4.33x, so the GPU run should land at "excellent" too (likely
     even higher, since the host-sync removal matters more on GPU).
3. Two traces get written: `results/hw2/trace_baseline.json` and
   `results/hw2/trace_optimized.json`. Open both at https://ui.perfetto.dev to
   look at the actual timelines.
4. **Write HW2 Q1–Q4** using the real trace + speedup numbers:
   - Q1: what dominates the *baseline* trace (look for the long run of growing
     matmuls + the `.item()` sync gap each step).
   - Q2: ablation table — baseline → +KV cache → +bf16 → (+compile if you add
     it). You can measure intermediate steps by temporarily calling
     `time_loop` on partial variants, or reason from the profiler tables already
     printed by `profile()`.
   - Q3: biggest single change (almost certainly KV cache) and why it matters
     for 128 short decode steps on a tiny model.
   - Q4: which optimizations cut memory traffic (KV cache, bf16) vs CPU/launch
     overhead (no `.item()`, compile) — and which mattered more here.

## 4. HW3 — `hw3_compile_cuda_graphs.ipynb`

1. Run **all cells top to bottom**.
2. Part 1 (`sweep_elementwise_times` / `estimate_launch_overhead_us`): check
   `results/hw3/launch_overhead.png` — same shape as the CPU run (flat at small
   n, climbing past some knee), but the absolute µs values reflect real GPU
   launch overhead.
3. Part 2 (`fixed_decode_step` + dynamo): confirm
   - `same math as the broken step       PASS`
   - `compiles with fullgraph=True       PASS`
   - `0 graph breaks                     PASS`
   Then check the "broken vs fixed" timing cell — this is the one that hung on
   the Mac. On GPU it should complete and print a real `X.Xx` speedup with
   `tag = "compiled, one graph"`.
4. Part 3 (`make_graphed_callable`) — **GPU-only, this is the centerpiece**:
   - Self-check should print:
     - `matches eager on fresh inputs      PASS`
     - `static buffers handled correctly   PASS`
   - The final benchmark cell compares eager / manual CUDA graph /
     `torch.compile(default)` / `torch.compile(reduce-overhead)` and writes
     `results/hw3/decode_step_latency.png`.
   - If `reduce-overhead` errors about overwritten outputs in the sequential
     loop, **that's expected and is the point of Q3** — don't try to work
     around it, just explain it in the writeup.
5. **Write HW3 Q1–Q3** using the real numbers/plot. Outline answers are in
   `asgmnt_plan.md` — Q1 (.item()/if/print and why .item() is doubly bad), Q2
   (why graph replay wins at batch 1, which constraints you respected), Q3 (two
   cases where `torch.compile` doesn't pay off, and how HW1/HW2's tools would
   detect each).

## 5. Final check before submitting

- [ ] All three notebooks executed top-to-bottom with no errors, fresh outputs
      visible (not stale/cleared)
- [ ] HW2 correctness == PASS and speedup tier recorded
- [ ] HW3 Part 3 self-check == PASS (not SKIP — SKIP means you're still on CPU)
- [ ] `results/hw1/`, `results/hw2/`, `results/hw3/` each contain their expected
      files (json/png/traces)
- [ ] All four writeup sections filled in with real numbers (no
      `*(your answer here)*` placeholders left)
- [ ] Submit the executed `.ipynb` files + the `results/` folders, due **June 27**
