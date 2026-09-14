# S1 checkpoint design (differential)

date: 2026-09-14 (S1 pre-requisite)

## 1. scheme: base + zstd-compressed diffs
- base: full fp32 (or active-precision) snapshot at run start / every N bases.
- diffs: per-tensor delta vs base, zstd-compressed. MoE experts change
  sparsely (top-k firing) → diffs stay small.
- zstd decision (S1 pre-requisite): vendor under third_party/zstd/
  (reproducible offline builds). Raw diffs are emergency fallback only.
  CI must build on all 3OS with vendored zstd.
- no silent format change: magic + version header (data_pack style).

## 1b. atomicity (mandatory)
- write to temp file → fsync → rename. Never write in place.
- keep 2 generations (latest + previous). Prune older only after the new
  generation verifies (checksum OK + resume dry-run OK where cheap).
- verification gate: save → resume → trajectory match (bit-identical
  single-thread, tol 1e-6 MT).

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
- RNG verification: save→resume→trajectory match must show zero RNG-related
  divergence. Covers dataloader shuffle, dropout (if any), routing noise.
  Any divergence → checkpoint feature blocked (S1 No-Go per plan §11).

## 5. budgets (S1 pre-requisite estimates)
- wall-clock: estimated from proxy rates before S1 launch; recorded in run
  manifest. Re-estimate if config changes.
- disk: checkpoint size (base + diffs) × 2 generations. Must fit
  artifacts/ volume with headroom; check before S1 launch.
- post-S1: checkpoints are either handed to S2 (documented run-id) or
  discarded. Reuse as init vs scratch is an S2-entry decision, recorded
  in the S1 exit manifest (no silent reuse).

## 6. verification gate
- save → resume → loss trajectory match (bit-identical single-thread,
  tol 1e-6 MT). Fails gate → checkpoint feature blocked, S1 continues
  without resume (fresh runs only).
