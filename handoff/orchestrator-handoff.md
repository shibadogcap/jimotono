# Orchestrator handoff (2026-09-14)

Reader: the next orchestrator. Read this first, then the docs in §3.
Do NOT run Stage 4a until the user reviews this document.

## 1. Philosophy

- Optimize relentlessly, compromise never — but honesty over optimism.
  Every number reported must state its regime (dims, threads, build flags).
- Ultimate goal: a smart agent on ordinary CPU+RAM+SSD.
- Hardware · model · engine are designed together. A kernel decision
  (e.g. T-MAC副線) is also a model decision (bits per sublayer) and a
  hardware decision (which SIMD backend runs it where).
- Stage gates G1–G4 (RULE.MD): G1 bit-identical (order-only changes),
  G2 statistical equivalence of function changes (final val ≤1%, drop
  excluded), G3 same-function optimization (bit or 1e-6), G4 numerically
  equivalent but routing-coupled divergence (structure + ≤1% + bounded
  amplification + perf). Never squeeze G4 criteria into G3.
- Proxy metrics are dangerous. toks/s across vocabs, train loss after
  saturation, T=1 dispatch, small-dim cycle counts — all lie in specific
  ways documented below. Always compare on same wall-clock or same bytes,
  val CE per byte normalized.

## 2. Non-obvious lessons (most valuable)

- **-mavx2欠落**: `train_bwd.c` silently ran scalar for months of tuning
  while neighbors were AVX2. Fix gave swiglu_bwd 3.8x. Lesson: verify
  SIMD is actually active (`__AVX2__` per TU, `nm`/disasm), never assume.
  Also: compiler FMA contraction voids source-level no-FMA policy —
  `-ffp-contract=off` is now global (MSVC `/fp:precise` covers it).
- **Amdahl再平衡**: after every speedup the optimum moves (12T→6T→2T).
  Regime baselines are stage-fixed; re-measure only between stages
  (RULE: threads {1,2,3,4,6} only, no fine re-search inside a stage).
- **proxy解像度限界**: d=64 proxy cannot resolve what d=1024 needs
  (factorized head +2.2% floor, lowbit +1%). Proxy最適 ≠ design最適.
  k/d ratio match (0.25) is necessary, not sufficient.
- **load balancing感度**: μ=0.1 looked dead at 20 steps, alive at 30+.
  Sensitivity needs the right window. Entropy mean hides instantaneous
  concentration — log per-expert max/min/median.
- **renormalize標準**: not renormalizing after drop changes dynamics
  MORE than renormalizing. Standard implementations renormalize. RULE.
- **train loss飽和**: relative diffs explode after saturation (279% on
  zeros). Gates use val loss only. Never judge on saturated train loss.
- **T=1は本番経路でない**: per-token dispatch never fires batching/drop/
  renormalize. T=1 agreements prove nothing about T=512. RULE.
- **ゲート混同**: drop rate is a §1.2 design trigger (steady-state only),
  not a G2 criterion. 5%-of-what and which-average must be explicit.
- **帯域律速は結果であって原因ではない**: demand roofline numbers
  (FLOPs/byte vs ridge) + csw/fork-join costs before concluding.
  LLC-thrash was a story; perf said miss rate 3.8–11.4% (rejected).
- **O_DIRECT fd discipline**: aligned buffer/len/offset or the kernel
  says EINVAL. Two CI failures came from unaligned test reads, not impl.
- **MSVC is a different language**: SEEK_SET needs stdio.h, no
  aligned_alloc (use _aligned_malloc), constant 1.0/0.0 is error C2124
  (use -INFINITY), C4996 warnings are noise. CI (8 jobs) is the only
  Windows machine we have.
- **Subagent bandwidth contention**: parallel load tests trash each
  other's numbers. Perf runs are single-process, orchestrator-sequential
  (RULE). Agents implement + correctness only.

## 3. Current state

- main = 583d47a (pushed, CI green 8/8 on 2da5231; re-verify after new pushes).
- Done: P2 engine, Phase D/D2-D7 (roofline: pure bandwidth-bound),
  Phase F/F2-F4 (100M proxy, held-out, names LM val 0.744, TinyStories
  eval), Phase G (MoE GEMM, G4 passed, 4.5-5.7x), P3a/b/c/d design docs,
  Stage 0 (AGENTS C1/C2/C6/C7, blockers), Stage 1 (factorized flag,
  gate open), Stage 2 (G1 fusion, bit-identical), Stage 3/3b-1 (W8A8,
  INT8 fast paths), Stage 3b-2 (lowbit fq+LUT, G2 marginal 1.0-1.3%),
  CI publish prep, S1 plan (final) + checkpoint design.
- Key docs: analysis/ (roofline, gemm-design, p3-architecture,
  p3-inference-engine, p3-training-scale, p3-integration, p3-s1-plan,
  s1-checkpoint-design, *_validation/check docs), results/bench/
  (dated logs), reviews/ (P2 mid/final/followup), ROADMAP.MD (P3/P4
  revised), RULE.MD (operational law — read all of it).
- Numbers that matter: 5M-proxy 1270 toks/s peak (2T), 100M-proxy
  33.4 toks/s, TinyStories small-model ~1300-16000 toks/s by config,
  V48588 val floor ~1.29-2.35 by scale, STREAM η≈0.47-0.50.

## 4. Pending

- S1 Go/No-Go (Stage 4e): 7 conditions in p3-s1-plan.md §12 + checkpoint
  gate (save→resume→match, failure = No-Go, no fallback).
- P3e dataset streaming: design done, implementation required before S2.
- Windows real-run: CI green covers build+logic only. IOCP paths are
  stubs/fread backend. Real IOCP + O_DIRECT-equivalent verification open.
- per-block scale (Stage 3b-3): recorded, non-blocking, post-S1 or parallel.
- factorized head design-scale validation: S1 mandatory gate (4-stage
  fallback: dense → k=512 → full-INT4 → P3a redesign).
- NEON real hardware: none so far (compile checks only).
- N100 box (`ssh ubuntu`): single dir `~/jimotono-bench` only. 11GB RAM,
  AVX2+AVX-VNNI (dpbssd #UDs — use dpbusd), liburing-dev installed.

## 5. Antipatterns

- Advancing past an unpassed gate.
- Confusing proxy optima with design optima.
- Comparing without re-measurement (flags, threads, data order).
- Leaving design docs behind ("申送り" ≠ solved).
- Confusing CI green with on-hardware verification (perf, ISA, IOCP).
- Calling wall-time-neutral "no effect" (unmanifested ≠ absent).
- Smoothing a >5% regression as "expected".
- Judging with train loss after saturation / token counts across vocabs.

## 6. First task for the next orchestrator

- Stage 4a: 1B model definition + data loader.
- Gates: accounting matches model1b (§1 of s1-plan), loader roundtrip
  bit-identical. S1 plan §§10-12 are the contract. Implement nothing
  beyond 4a scope. One variable at a time. Report before proceeding.
