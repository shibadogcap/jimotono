# Windows I/O risks: IOCP + FILE_FLAG_NO_BUFFERING (random 4K)

- Premise: `include/jimotono/io_batch.h`, `src/io_batch.c` (Phase 1 `_lseeki64+_read` buffered fallback),
  `src/io.c` (Windows `jt_io_win_*` NOSUP stub), `include/jimotono/io_direct.h` / `src/io_direct.c`
  (Linux `O_DIRECT` real path), `analysis/p3-io-abstraction.md` §2.3, `DESIGN.MD` §3, AGENTS.md §1/§8.
- Scope: document only. No implementation, no build/run in this wt.
- Target path: S1 1B expert streaming I/O (COLD delta rows / low-frequency routed experts).

## 1. S1 1B expert streaming I/O path (baseline)

- S1 config (from `analysis/p3-training-scale.md` §3): L=24, d=1024, E=64/layer,
  expert_hidden=128, shared=2, top-k=4, GDN 5:1, total 0.712B, active 96.018M (<100M),
  dense INT4 69.6MiB (<2GB), SSD experts 264MiB.
- Fetch estimate (same source): ~16MiB/token (24 layers x top-4 x 172KiB), inside
  the 103MiB/token budget (AGENTS.md §8). Unmeasured elements remain; S1 exit gate is
  `fetch<=103MiB` measured + `drop steady-state<=5%` + `eta` recorded.
- HOT/COLD split (`io_batch.h`, `p3-io-abstraction.md` §1.4):
  HOT (shared experts / attention projections / frequent routed) = pin + LRU + buffered,
  `O_DIRECT` forbidden. COLD (delta 15-18% incoming rows) = `runs_compress` run-ified
  short sequential run bundle, page-cache bypass on Linux (`O_DIRECT` + `io_uring`).
- Windows Phase 1 real path: buffered `_lseeki64+_read` loop, single-thread 1C1T
  (fd offset moves, so fd sharing is forbidden). `OVERLAPPED`/IOCP is Phase 2 TODO,
  currently NOSUP stub (`src/io.c`).

## 2. FILE_FLAG_NO_BUFFERING risks (random 4K)

### 2.1 Sector alignment 512B / 4KB

- Constraint: buffer must be sector-aligned (`_aligned_malloc`), and both `length`
  and `offset` must be sector multiples. Sector size is volume-dependent
  (512e vs 4K native); obtain at startup via `GetDiskFreeSpace` (cf. `p3-io-abstraction.md` §2.3).
- Risk: expert streaming I/O is a bundle of short runs (`runs_compress` output, often
  4K-order). Any non-multiple tail needs **round-up padding read + caller-side truncation**
  (P3 decision item). Padding inflates bytes read per token and eats the 103MiB/token
  budget. Linux `O_DIRECT` has the same shape (512 required / 4096 recommended,
  fail-closed `JT_ERR_INVAL+EINVAL`), but Windows violation cost is higher
  (`ERROR_INVALID_PARAMETER`), so pre-check (`jt_io_direct_is_aligned` equivalent with
  variable align) is mandatory before issuing I/O.
- Risk: FAT32/exFAT or some network FS do not support `NO_BUFFERING`; need buffered
  re-open fallback (same shape as Linux open-fallback on `EINVAL/EOPNOTSUPP/ENOSYS`).
  Detection must be at open time; `pread`-side re-fallback on the same fd is impossible.

### 2.2 Completion order not guaranteed (IOCP)

- `OVERLAPPED` + `GetQueuedCompletionStatus` is the `io_uring` CQE counterpart, but
  completions return **out of order**. The `jt_io_pread_batch` contract (ordered specs,
  short-read retry, EOF-before-end = EIO, `len==0` no-op) must be re-established in the
  completion-drain layer (per-req index tracking, in-window retry folded into
  window capacity 256). Current uring path (`io_batch_uring.c`) already folds retries
  into the window; the Windows backend must copy that discipline in a separate function
  (do not mix into the `_read` loop, per existing TODO).
- Risk: single-thread 1C1T premise breaks if completions are awaited with blocking
  `GetQueuedCompletionStatus` on the decode critical path. Design rule
  (`p3-io-abstraction.md` §5): never place blocking I/O on the decode path;
  `submit` -> dense-prefix compute -> `wait` overlap with a fixed inflight cap (256).
  IOCP thread-pool or alertable-wait introduction would violate the no-new-thread rule
  unless the P3 owner explicitly revises it.

### 2.3 O_DIRECT differences (Linux vs Windows)

| Aspect | Linux `O_DIRECT` | Windows `FILE_FLAG_NO_BUFFERING` |
|---|---|---|
| Strength | Per-I/O bypass enforced; misaligned I/O errors (fail-closed detectable) | Per-handle flag; stronger than macOS `F_NOCACHE` but still handle-scoped, sector-multiple enforced |
| Alignment | 512 required, 4096 recommended (`posix_memalign`) | Sector size variable (512e/4K native); buffer/offset/length all multiples, `_aligned_malloc` |
| Tail handling | 512-aligned pre-check, chunk split for `UINT_MAX` oversize | Non-multiple tail needs padded read + truncation (budget impact) |
| Unsupported FS | open errno -> buffered fallback (explicit) | Re-open without flag (explicit); must mirror Linux table |
| Async pairing | `io_uring` (buffered fd today; direct+uring kept separated) | IOCP (`OVERLAPPED`); must be a separate function from `_read` loop |

- Consequence: Windows COLD="cache-pollution zero" is achievable in principle (unlike
  macOS `F_NOCACHE` best-effort), but only after the alignment-variable check, tail
  policy, and fallback matrix are frozen (cf. `p3-io-abstraction.md` §4.1 items 3-4).

## 3. Random-4K performance risks

- SSD random-read latency (20-100us, `DESIGN.MD` §3.3) is near-infinite for a CPU;
  blocking I/O is never allowed on decode. S1 COLD is ~16MiB/token of short runs;
  without prefetch overlap, per-layer `_lseeki64+_read` round-trips serialize decode.
- `NO_BUFFERING` bypasses the page cache, so HOT reuse (shared experts, frequent routed)
  must stay buffered/pinned; applying `NO_BUFFERING` to HOT would turn hits into
  repeated random 4Ks. Keep the HOT/COLD fd separation strict.
- Unaligned or sub-sector tails cause read-modify-write amplification at the device
  layer on 4K-native volumes; co-locate gate/up/down + LUT + scale in the same page
  (`DESIGN.MD` §5.1) and keep run lengths sector multiples where the exporter controls layout.
- Measurement: do not claim Windows numbers equal Linux numbers. Record submit counters
  (inflight, wait time, pin-hit rate, MiB/token) per OS; <5% deltas are noise
  (`RULE.MD` §6 convention).

## 4. Impact prediction on S1 1B

- Phase 1 (current, buffered `_lseeki64+_read`): functionally correct; performance risk is
  serialized random 4Ks on the decode path. Mitigation is caller-side lookahead window
  (S-U-L union + bounded overfetch, router-lookahead 1 layer ahead) + HOT pin, not
  backend change. No budget breach is predicted from the backend itself because the
  ~16MiB/token estimate has ~6x headroom to 103MiB/token, but **unmeasured** (sync growth,
  wait mixing) — S1 exit measurement decides.
- Future `NO_BUFFERING` without tail policy: padding could inflate COLD bytes by up to
  ~1 sector per short run in the worst case (run count dependent). With hundreds of runs
  per token this is still inside 103MiB/token on paper, but must be measured via the
  submit counter, not assumed.
- Future IOCP without ordering layer: correctness risk (misattributed completions) exceeds
  performance risk. Gate: separate-function IOCP + ordered drain + in-window retry +
  `JT_ERR_INVAL` pre-check, with the 3-OS same-pack consistency test
  (`p3-io-abstraction.md` §4.2) green before any S1 perf claim.

## 5. Gate (before S1 claims Windows parity)

1. Freeze variable-align check + tail pad/truncate policy + fallback matrix (P3 owner).
2. Keep Phase 1 buffered path as the correctness baseline; IOCP as a separate function only.
3. 3-OS same-pack consistency test green (correctness gate; speed is reference-only).
4. Report MiB/token + wait/pin-hit per OS; do not merge Windows numbers with Linux numbers.

## References

- `include/jimotono/io_batch.h`, `src/io_batch.c`, `src/io.c`, `src/io_direct.c`
- `analysis/p3-io-abstraction.md` §§1-5, `analysis/p3-training-scale.md` §3
- `DESIGN.MD` §3/§5, AGENTS.md §1/§8
