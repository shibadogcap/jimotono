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

## 4. factorized head × lowbit: S1 composite gate
- factorized単独: val diff ≤ 1% @2000 steps (design scale)
- 低ビット単独: val diff ≤ 1% (TinyStories T=512。真LUT +0.98%で通過)
- 複合 (factorized + 低ビット): val diff ≤ 1%
- 複合未達時のフォールバック順序:
  1. 低ビットをINT8に戻す (T-MAC断念、W8A8本線維持)
  2. factorized headをk=512へ
  3. full head INT4へ
  4. 未達時はP3a再設計
- proxy解像度注記: d=64とd=1024の量子化感度差仮説あり（大dほど鈍感）。
  S1でdesign scale実測を必須とする。
- details: analysis/p3a-factorized-head-validation.md、
  analysis/p3b-tmac-gradient-check.md

## 5. S1 verification items
1. 1B training fits 16GB RAM
2. factorized val diff ≤ 1%
3. throughput (steps/s, toks/s)
4. memory (peak RSS + budget reconciliation)
5. grad norm stability

## 6. Go/No-Go
- Go: all 5 green + factorized gate (or fallback stage passed)
- No-Go: any red → S2/S3 blocked, design review
