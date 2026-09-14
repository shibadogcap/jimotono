# CI coverage (run 34836936369, 8/8 success)

date: 2026-09-14
commit: 2da5231

## jobs (all 20/20 pass)
- linux-gcc, linux-clang, linux-arm64-gcc: pass
- macos-clang, macos15-x64-clang, macos14-arm64-clang: pass
- windows-msvc, windows-arm64-msvc: pass (20/20 incl. io_xpack/data_pack via fread backend)

## I/O tests
- io_xpack/io_direct/uring run everywhere; platform-specific paths are SKIP-green
  outside their OS (uring non-Linux, direct non-Linux). Windows exercises the
  fread backend + stubs (IOCP still NOSUP stub).
- IOCP real-run: NOT covered (no IOCP implementation yet).

## instruction dispatch
- Compile-time guards only (AVX2/NEON/VNNI/AVX512 ifdefs). CI verifies the
  guards compile per platform; runtime dispatch logging is out of scope.
- AVX-VNNI (N100-only) and AVX512-VNNI (Ryzen-only) are NOT executed in CI.

## limits (what CI is NOT)
- Build + logic correctness only. NOT performance, NOT instruction execution,
  NOT IOCP/async paths, NOT 1B-scale runs.
