# S1 implementation plan (final, pre-implementation)

date: 2026-09-14

## 1. model config (model1b)
- total 0.712B, active 96M, dense INT4 69.6MiB, SSD experts 264MiB
- fetch ~16MiB/token (< 103MiB budget)

## 2. data
- TinyStories 16MB subset (vendored, no network). pack regenerable
  (data/tinystories16M.jtdp via --write-pack).

## 3. tokenizer
- V=48,588 llm-jp v2.2 Unigram, C implementation (src/bpe.c).
- vocab file: data/llmjp-v22.spm (vendored) → data/llmjp-v22.jtvocab
  (regenerable via scripts/bpe_make_vocab.py, gitignored).

## 4. engine scale-up (d=1024, L=24, E=96, top-4, shared=2)
- model1b accounting defines the target; engine kernels are dim-agnostic
  (JT_BWD_MAX_WIDE bounds apply — verify 1024/48588 fit before S1).

## 5. factorized head (opt-in, composite gate)
- §7 composite gate applies.

## 6. lowbit (W8A8 main line, INT4/INT2副線)
- W8A8 default-off path + INT2/INT4 lowbit paths (mode 1/2).
- S1 uses fp32 default; lowbit enabled per gate stages.

## 7. I/O (3OS)
- jt_io_submit/poll/wait thin layer. Linux O_DIRECT+uring real-run,
  macOS F_NOCACHE+kqueue readiness, Windows fread backend.
- 3OS identical-pack test (io_xpack) green in CI.

## 8. logging / reproducibility
- throughput (steps/s, toks/s), memory (RSS + budget), grad norm per step.
- deterministic seeds (single-thread bit-identical; MT order documented).
- val CE per byte normalized (RULE).

## 9. engine checklist for 1B
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

## 10. S1 stages (gated, no bulk implementation)
- Stage 4a: 1B model definition + data loader. gate: accounting matches §1,
  loader roundtrip bit-identical.
- Stage 4b: engine 1B readiness (dims fit, 16GB budget assert). gate: ctest
  + proxy equivalence at scale dims.
- Stage 4c: factorized head enable. gate: §7 composite (factorized part).
- Stage 4d: lowbit enable. gate: §7 composite (lowbit part).
- Stage 4e: full evaluation, 2000 steps. gate: §7 composite + §11 items.

## 11. Go/No-Go (final)
- Go conditions:
  1. 1B training completes within 16GB RAM
  2. factorized single ≤ 1%, composite ≤ 1%
  3. lowbit single ≤ 1%, composite ≤ 1%
  4. throughput ≥ 100 tok/s
  5. RSS < 14GB
  6. grad norm: no divergence
  7. checkpoint save→resume→trajectory match
- Fallback per condition: §7 order (INT8 revert → k=512 → full INT4 → P3a).
  Any red without fallback pass → No-Go, S2/S3 blocked, design review.
