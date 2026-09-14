# S1 preparation plan (1B establishment)

date: 2026-09-14

## 1. model config (model1b)
- total 0.712B, active 96M, dense INT4 69.6MiB, SSD experts 264MiB
- fetch ~16MiB/token (< 103MiB budget)

## 2. data
- TinyStories 16MB subset (vendored, no network). pack regenerable.

## 3. engine checklist for 1B
- [x] MoE GEMM path (Step 4 micro, G4 passed)
- [x] renormalize + cap 1.5 + aux loss (drop steady-state 3.34%)
- [x] W8A8 path (default off, +0.28%)
- [x] INT8 fast paths (AVX2/VNNI)
- [x] factorized head flag (gate open, dense default)
- [x] x-platform I/O thin layer (Linux real-run, macOS real-run, Windows fread)
- [ ] Stage 3b-2 INT4/INT2 + T-MAC (design approved, impl pending)
- [ ] G1 fusion decode remeasure (after decode impl)

## 4. factorized head: S1 mandatory gate (4-stage fallback)
- G0: dense 1B monotonic val decrease
- G1: factorized val diff ≤ 1% @2000 steps (design scale)
- fallback: k=512 → full head INT4 → P3a redesign
- details: analysis/p3a-factorized-head-validation.md

## 5. S1 verification items
1. 1B training fits 16GB RAM
2. factorized val diff ≤ 1%
3. throughput (steps/s, toks/s)
4. memory (peak RSS + budget reconciliation)
5. grad norm stability

## 6. Go/No-Go
- Go: all 5 green + factorized gate (or fallback stage passed)
- No-Go: any red → S2/S3 blocked, design review
