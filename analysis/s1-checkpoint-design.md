# S1 checkpoint design (differential)

date: 2026-09-14 (S1 pre-requisite)

## 1. scheme: base + zstd-compressed diffs
- base: full fp32 (or active-precision) snapshot at run start / every N bases.
- diffs: per-tensor delta vs base, zstd-compressed. MoE experts change
  sparsely (top-k firing) → diffs stay small. (zstd is external dep:
  vendoring vs system lib decision is S1-impl task. fallback: raw diffs.)
- no silent format change: magic + version header (data_pack style).

## 2. interval
- every 500 steps or 30 minutes, whichever first.
- emergency save on signal (SIGTERM/SIGINT handler sets flag, checked
  between steps; never mid-kernel).

## 3. location
- artifacts/ckpt/<run-id>/ (gitignored, manifest committed per ROADMAP §4.3).
- run-id: YYYYMMDD-<short-hash>. manifest records config hash + bench.

## 4. resume: --resume <run-id>
- loads base + latest diff, verifies checksum, continues step counter +
  optimizer state (optim8 int8+m/v scales included).
- RNG state: deterministic re-seed from step counter (documented;
  MT order effects noted where applicable).

## 5. verification gate
- save → resume → loss trajectory match (bit-identical single-thread,
  tol 1e-6 MT). Fails gate → checkpoint feature blocked, S1 continues
  without resume (fresh runs only).
