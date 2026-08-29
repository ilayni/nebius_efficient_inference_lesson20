# Submission feedback — `roofline_to_cuda_ilay_nisim_boris_kogan.zip`

Reviewed: HW1 (roofline), HW2 (decode optimization), HW3 (compile & CUDA graphs),
plus the `results/` artifacts. Reviewed against the actual submitted notebooks
(GPU outputs embedded) and the JSON/PNG artifacts in the zip.

**Overall:** strong, complete submission. All three notebooks run top-to-bottom,
every self-check PASSes, HW2 hits 4.62× (excellent tier), HW3 hits 3.43× with a
hand-rolled CUDA graph, and no writeup placeholders remain. The HW1 writeup numbers
all reconcile exactly with `results/hw1/roofline_data.json`. The items below are
about a few **factual/quantitative slips in the writeups** — nothing blocks running,
but a couple could cost points on a writeup-graded rubric. It's still pre-deadline
(due June 27), so all of this is fixable before you submit.

---

## GRADE

> The notebooks carry **no numeric rubric** — only pass/fail self-checks and HW2's
> explicit speedup tiers. The scheme below is constructed for a number: each HW is
> 100 pts split **Implementation & self-checks (50) · Results/performance (20) ·
> Writeup (30)**, the three HWs weighted equally.

| HW | Impl & self-checks | Results | Writeup | Total (as-submitted) | After fixes |
|---|---|---|---|---|---|
| HW1 — roofline | 50/50 | 20/20 | 20/30 *(Q1 misdiagnosis, see C5)* | **90** | **99** |
| HW2 — decode opt | 50/50 | 20/20 *(4.62×, excellent)* | 27/30 *(Q2 ablation, see N1)* | **97** | **97** |
| HW3 — compile/graphs | 50/50 | 20/20 *(3.43× manual graph)* | 30/30 | **100** | **100** |
| **Overall (equal weight)** | | | | **≈ 96 / 100** | **≈ 99 / 100** |

**As-submitted: ~96/100.** After the C1–C4 writeup fixes (already applied in a prior
pass) **and the C5 fix applied 2026-06-19**, the corrected submission is **~99/100**.
The only remaining sub-30 writeup is HW2 Q2 (N1, optional polish — the ablation rows
aren't reproduced in the executed notebook, though they're now honestly flagged).

---

## STATUS — five CRITICAL items fixed (2026-06-19)

All C1–C4 writeup fixes were applied to the HW2/HW3 notebooks in **all three
locations** (`gpu/*_gpu.ipynb` source, root `*.ipynb`, `submission/*.ipynb`),
the root `.md` mirrors were regenerated, and the submission zip was rebuilt.
**C5 (HW1 Q1) was applied the same way on a second pass** — patched in all three
HW1 notebooks, `hw1_roofline.md` regenerated, zip rebuilt.
Only writeup markdown cells changed — no code, outputs, or `results/` artifacts
were touched, so the 4.62× / 3.43× headline results and self-check PASSes stand.

| Item | Status | What changed |
|---|---|---|
| C1 — ablation baseline | ✅ Fixed | Speedups recomputed vs 214.9 ms (1.63× / 1.40× / 2.54×); added a caveat that the 3 intermediate rows came from a separate session; cross-reference in HW3 Q3 updated |
| C2 — fabricated sync cost | ✅ Fixed | Dropped the 0.24 ms / 31 ms figures; reframed as lost CPU/GPU overlap, citing the measured ≈6 µs launch cost |
| C3 — HW3 kernel arithmetic | ✅ Fixed | ~16 kernels/96 µs → ~22 kernels/~130 µs + ~50 µs compute, now reconciling to 180.6 µs |
| C4 — Q1/Q3 O(n²) framing | ✅ Fixed | Q3 "negligible" → "small FLOP savings; 1.63× is mostly sync removal; second-order vs launch overhead" |
| C5 — HW1 Q1 eager points | ✅ Fixed | Eager `compute-K` relabeled **memory-bound** (not "launch-overhead bound"); explained the plot floats only because it uses the ideal load-once AI while real AI stays ≈0.25; this also resolves N2 (compiled K=64 now "climbing the BW slope toward memory-bound") |

Nice-to-have items below were **not** applied (optional polish).
Note: the team's working notes (`asgmnt_ongoing.md`, `gpu/gpu_summary.md`) still
carry the old numbers — they're scratch docs, not graded, so left as-is.

---

## CRITICAL (fix before submitting) — ✅ ALL FIXED (see Status above)

### C1 — HW2 Q2 ablation table: speedups computed against the wrong baseline · ✅ FIXED
The table header says "vs baseline" and the notebook's printed baseline is
**214.9 ms**. But three of the five rows were computed against a **~180 ms**
baseline (from a different run), so the speedup column is internally inconsistent:

| Row | Time | Stated | Actual (vs 214.9 ms) |
|---|---|---|---|
| + KV cache | 131.5 ms | 1.37× | **1.63×** |
| + KV + bf16 | 153.4 ms | 1.18× | **1.40×** |
| + KV + bf16 + compile(dynamic) | 84.6 ms | 2.13× | **2.54×** |

Rows 1 (1.00×) and 5 (4.62×) *do* match 214.9 ms — only the middle three are off.
Check: 131.5 × 1.37 = 180, 153.4 × 1.18 = 181, 84.6 × 2.13 = 180 — all three imply
the same ~180 ms baseline, not 214.9 ms. **Fix:** recompute every row against one
baseline. This also strengthens your argument — KV cache alone is 1.63×, not 1.37×.

### C2 — HW2 Q1: the "each sync ≈ 0.24 ms → ~31 ms" claim is unsupported and self-contradictory
You state each `.item()` host sync "costs ≈ 0.24 ms." Nothing in the submission
measures that, and it contradicts your own HW3 Part 1 result (per-launch overhead
= **5.98 µs**). A sync's intrinsic cost is microseconds; what actually hurts is the
**lost CPU/GPU overlap** — the CPU stalls until the GPU drains, serializing the two.
**Fix:** drop the fabricated 0.24 ms / 31 ms figures and frame the cost as lost
overlap, or cite a number you actually measured.

### C3 — HW3 Q2: "≈16 kernels ≈ 96 µs" doesn't reconcile with 180.6 µs
You write that ~16 kernels × 6 µs ≈ 96 µs "matches the measured 180.6 µs/step."
96 ≠ 180. Your own data implies the real count: eager 180.6 µs → manual graph
52.6 µs means replay removed ≈ 128 µs ≈ **~21 launches**, so it's closer to ~22+
kernels/step, not 16. **Fix:** use ~22 kernels (LayerNorm is several kernels, each
Linear is matmul + bias, plus GELU and the residual add, × 4 blocks), which makes
the arithmetic actually land near 180 µs.

### C4 — HW2 Q1 vs Q3: contradictory framing of the O(n²) recompute
Q1 lists the no-KV-cache O(n²) recompute as a "dominant symptom"; Q3 calls it
"negligible." With the corrected C1 numbers, KV cache saves ~39% of wall time —
not negligible. **Fix:** pick one consistent story (e.g. "real but second-order vs
launch overhead on this toy model") and align both answers.

### C5 — HW1 Q1: eager `compute-K` points mislabeled "launch-overhead bound" · ✅ FIXED
The original answer called the flat eager `compute-K` series **launch-overhead
bound** ("~6 µs/launch dominates the wall time"). That's wrong — and it contradicts
the assignment's own Part 1b text ("each step is a separate kernel with its own HBM
round-trip"). The arithmetic: eager K=64 at 0.65 TFLOP/s ⇒ (2·64·2²⁴)/0.65e12 ≈
3300 µs ⇒ 51.6 µs/step. Each step is 2 kernels (mul, add), so **≈ 25.8 µs/kernel —
exactly the measured `elementwise` cost** (67 MB ÷ 2598 GB/s). Launch overhead
(≈ 6 µs) is only ~23% of that; **memory traffic is ~77%**. The points only *appear*
to float below both ceilings because they're plotted at the **ideal load-once AI**;
at their true AI (≈ 0.25, constant in K, since bytes scale with K) they sit on the
BW slope at ~77% — **memory-bound**. The student correctly used launch-bound
reasoning in HW2/HW3 (batch-1, tiny tensors) but misapplied it to HW1's 16 M-element
ops. **Fix applied:** rewrote the Q1 framing sentence, the eager paragraph
(→ memory-bound, with the round-trip explanation), and the compiled paragraph
(fusion removes the K−1 intermediate HBM round-trips → AI rises → "climbing the BW
slope toward memory-bound" at 47%, which also resolves N2).

---

## NICE TO HAVE (polish / extra depth)

### N1 — HW2 Q2: the ablations aren't reproduced in the executed notebook
The notebook only runs baseline + final config; the four intermediate rows are
typed in from separate runs. The assignment says "measure them one at a time" —
adding the ablation cells (or a short note that they were measured in separate runs
on the same seed) makes the table self-evidencing. Also, row 2 bundles "KV cache +
.item() removed," so it doesn't isolate the two as cleanly as the "one at a time"
instruction asks.

### N2 — HW1 Q1: "compiled K=64 is memory-bound" is a slight stretch · ✅ RESOLVED BY C5
At AI=32 the BW ceiling is 32 × 3.35 = 107 TFLOP/s and you achieved 50.6 (47%).
That's *approaching* the slope, not sitting on it. The C5 rewrite now phrases this
as "climbing the BW slope toward memory-bound" (47%), so this is handled.

### N3 — HW3: explain why `torch.compile` (default) gave only 1.01×
The benchmark shows default compile at 1.01× vs eager — worth one sentence: default
mode fuses kernels but does **not** capture CUDA graphs, so at batch 1 the per-step
launch count (and thus launch overhead) is essentially unchanged; only
`reduce-overhead` / manual graphs remove the launches.

### N4 — HW2 Q4: you argue past the prompt's premise (this is good, just flag it)
The prompt asserts the loop "is memory-bandwidth-bound"; you argue it's actually
launch-bound for this toy model. That's correct and well-supported — just make the
move explicit ("the premise holds for production 7B+ models; for this 2-layer toy it
doesn't, and here's the evidence") so it reads as a deliberate refinement, not a
contradiction. You partly do this already; one framing sentence seals it.

### N5 — HW3 Q1: scope creep into HW2
Q1 is about a single `broken_decode_step`, but the answer references "a 128-step
decode loop ... 128 forced stalls." Fine as color, but tighten it to the single-step
question being asked.

---

## What's clearly right (keep as-is)
- HW1: all roofline points below the roof, matmul AI = N/3 derivation correct,
  measured 341:683:1365 = 1:2:4 ratio nailed, numbers match the JSON exactly.
- HW2: `optimized_loop` reproduces baseline tokens exactly (fp32 PASS); StaticCache
  + `reduce-overhead` + the `.clone()` rationale is correct and well explained; 4.62×.
- HW3: per-launch overhead sweep + knee location is clean; `fixed_decode_step` uses
  `torch.where` to keep the branch on-device (0 graph breaks); manual CUDA-graph
  capture (side-stream warmup, static buffers, `copy_` in / `clone` out) is textbook,
  and you correctly let `reduce-overhead` fail and explained it rather than hacking
  around it.
- Packaging: zip matches the `submission/` tree, `trace_check.json` correctly
  excluded, all `results/` artifacts present, no leftover placeholders.
