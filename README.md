# Efficient Inference — Lesson 20

Homework for the efficient-inference lesson of the Nebius ML course, covering
roofline analysis, decode-path optimization, and `torch.compile` / CUDA graphs.

## Contents

| Path | What it is |
|---|---|
| `hw1_roofline.ipynb` | HW1 — roofline model and arithmetic-intensity analysis |
| `hw2_decode_optimization.ipynb` | HW2 — decode-step latency optimization |
| `hw3_compile_cuda_graphs.ipynb` | HW3 — `torch.compile` and CUDA graph capture |
| `gpu/` | GPU-executed variants of the notebooks, plus result artifacts |
| `submission/` | Final submitted notebooks |
| `*.md` | Instructions, working notes, plan and feedback |

## Setup

```bash
cp .env.example .env   # then fill in your tokens
```

GPU runs were done on a remote Nebius instance; host details in the notes are
replaced with `<GPU_HOST>` / `<SSH_USER>` placeholders.

`.env` is git-ignored; never commit real credentials.
