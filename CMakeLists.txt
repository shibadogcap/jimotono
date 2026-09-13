cmake_minimum_required(VERSION 3.20)
project(jimotono VERSION 0.0.1 LANGUAGES C)

# C11, no extensions for portability (Linux/macOS/Windows)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Default to Release if unspecified (bench reproducibility)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Warnings: strict but portable across clang/gcc/MSVC
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)
endif()

# SIMD is opt-in per TU via ifdef guards, never forced globally.
# x86_64: -mavx2 / -mavx512* must be set per-target, not globally,
# to keep the default binary runnable on N100/i5-8th baseline.
# ARM64: NEON is baseline on AArch64, no flag needed.

option(JIMOTONO_BUILD_BENCH "Build microbenchmarks" ON)
option(JIMOTONO_NATIVE "Build with -march=native (bench only, never for release)" OFF)
option(JIMOTONO_USE_URING "Enable Linux io_uring batch-read via liburing (default OFF, fallback ENOSYS)" OFF)

if(JIMOTONO_NATIVE AND NOT MSVC)
  add_compile_options(-march=native)
endif()

# ---- core library ----
add_library(jimotono_core
  src/arena.c
  src/tmac.c
  src/gdn2.c
  src/routing.c
  src/io_batch.c
  src/io_batch_uring.c
  src/io_direct.c
  src/train_bwd.c
  src/checkpoint.c
  src/optim8.c
  src/esmoe.c
  src/mixed_prec.c
  src/lut_aware.c
  src/model1b.c
  src/moe_layer.c
  src/data_pack.c
)
# POSIX宣言 (setenv/fileno/mkdtemp等) は厳格C11(__STRICT_ANSI__)で隠れる。
# Linuxのみ全ターゲットへ_GNU_SOURCE (O_DIRECT用定義と統一)。
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_compile_definitions(_GNU_SOURCE)
endif()
target_include_directories(jimotono_core PUBLIC include)
# O_DIRECT (io_direct.c) はglibcで_GNU_SOURCEを要する。上記で充足。
# io_uring (Linuxのみ, Phase 2足場)。既定OFFでfallback (ENOSYS) ビルド。
# 有効時のみ <liburing.h> をincludeしliburingにリンクする。
# O_DIRECTは本タスク範囲外 (buffered fd前提)。
if(JIMOTONO_USE_URING)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(WARNING "JIMOTONO_USE_URING is Linux-only; ignoring on ${CMAKE_SYSTEM_NAME} (fallback ENOSYS)")
  else()
    find_path(LIBURING_INCLUDE_DIR NAMES liburing.h DOC "liburing headers")
    find_library(LIBURING_LIBRARY NAMES uring DOC "liburing library")
    if(LIBURING_INCLUDE_DIR AND LIBURING_LIBRARY)
      target_compile_definitions(jimotono_core PUBLIC HAVE_LIBURING)
      target_include_directories(jimotono_core PRIVATE ${LIBURING_INCLUDE_DIR})
      target_link_libraries(jimotono_core PRIVATE ${LIBURING_LIBRARY})
      message(STATUS "io_uring enabled: ${LIBURING_LIBRARY}")
    else()
      message(FATAL_ERROR "JIMOTONO_USE_URING=ON but liburing not found (install liburing-dev). Fallback build uses -DJIMOTONO_USE_URING=OFF.")
    endif()
  endif()
endif()
# tmac/gdn2/routing use round()/isfinite()/exp() from <math.h>; libm on Unix.
if(UNIX)
  target_link_libraries(jimotono_core PUBLIC m)
endif()

# ---- bench common harness (ROADMAP 5.1/5.2: median+MAD, monotonic clock) ----
# NOTE: bench/common は測定基盤のみ。src/*.c と common.h には手を出さない。
add_library(jimotono_bench_common
  bench/common/bench_common.c
)
target_include_directories(jimotono_bench_common PUBLIC include bench/common)
if(UNIX)
  target_link_libraries(jimotono_bench_common PUBLIC m)
endif()

# ---- microbenchmarks (ROADMAP 5.2: harness + cases + metric) ----
if(JIMOTONO_BUILD_BENCH)
  add_executable(bench_gdn2 bench/kernel/gdn2.c)
  target_link_libraries(bench_gdn2 PRIVATE jimotono_core)

  add_executable(bench_tmac bench/kernel/tmac_lut.c)
  target_link_libraries(bench_tmac PRIVATE jimotono_core)

  add_executable(test_arena bench/kernel/test_arena.c)
  target_link_libraries(test_arena PRIVATE jimotono_core)

  add_executable(test_routing bench/kernel/test_routing.c)
  target_link_libraries(test_routing PRIVATE jimotono_core)

  add_executable(test_io bench/kernel/test_io.c)
  target_link_libraries(test_io PRIVATE jimotono_core)

  add_executable(bench_smoke bench/kernel/bench_smoke.c)
  target_link_libraries(bench_smoke PRIVATE jimotono_bench_common)

  add_executable(test_uring bench/kernel/test_uring.c)
  target_link_libraries(test_uring PRIVATE jimotono_core)

  add_executable(test_io_direct bench/kernel/test_io_direct.c)
  target_link_libraries(test_io_direct PRIVATE jimotono_core)

  add_executable(test_train_bwd bench/kernel/test_train_bwd.c)
  target_link_libraries(test_train_bwd PRIVATE jimotono_core)

  add_executable(test_optim_offload bench/kernel/test_optim_offload.c)
  target_link_libraries(test_optim_offload PRIVATE jimotono_core)

  add_executable(test_mixed_lut bench/kernel/test_mixed_lut.c)
  target_link_libraries(test_mixed_lut PRIVATE jimotono_core)

  add_executable(test_e2e_pipe bench/kernel/test_e2e_pipe.c)
  target_link_libraries(test_e2e_pipe PRIVATE jimotono_core)

  add_executable(test_cycle_bench bench/kernel/test_cycle_bench.c)
  target_link_libraries(test_cycle_bench PRIVATE jimotono_core jimotono_bench_common)
  add_executable(test_moe_layer bench/kernel/test_moe_layer.c)
  target_link_libraries(test_moe_layer PRIVATE jimotono_core)
  # bench_schema.md の commit 記録用にビルド時ハッシュを埋め込む (取得不可時は空)。
  execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE JIMOTONO_BUILD_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(JIMOTONO_BUILD_COMMIT)
    target_compile_definitions(test_cycle_bench PRIVATE JIMOTONO_BUILD_COMMIT="${JIMOTONO_BUILD_COMMIT}")
  endif()

  add_executable(test_model1b bench/kernel/test_model1b.c)
  target_link_libraries(test_model1b PRIVATE jimotono_core)

  add_executable(test_data_pack bench/kernel/test_data_pack.c)
  target_link_libraries(test_data_pack PRIVATE jimotono_core)

  # pack_jsonl: 空白区切り整数テキスト→.jtdp変換ツール (実行バイナリのみ。
  # ctestには追加しない。JSONL本格パーサは禁止=トークナイザ確定前の暫定)。
  add_executable(pack_jsonl bench/data/pack_jsonl.c)
  target_link_libraries(pack_jsonl PRIVATE jimotono_core)

  # train_longrun: 長時間学習ベンチ (実行バイナリのみ。ctestには追加しない)。
  add_executable(train_longrun bench/kernel/train_longrun.c)
  target_link_libraries(train_longrun PRIVATE jimotono_core)
endif()

enable_testing()
if(JIMOTONO_BUILD_BENCH)
  add_test(NAME arena COMMAND test_arena)
  add_test(NAME tmac COMMAND bench_tmac)
  add_test(NAME gdn2 COMMAND bench_gdn2)
  add_test(NAME routing COMMAND test_routing)
  add_test(NAME io_batch COMMAND test_io)
  add_test(NAME bench_smoke COMMAND bench_smoke)
  add_test(NAME uring COMMAND test_uring)
  add_test(NAME io_direct COMMAND test_io_direct)
  add_test(NAME train_bwd COMMAND test_train_bwd)
  add_test(NAME optim_offload COMMAND test_optim_offload)
  add_test(NAME mixed_lut COMMAND test_mixed_lut)
  add_test(NAME e2e_pipe COMMAND test_e2e_pipe)
  add_test(NAME cycle COMMAND test_cycle_bench)
  add_test(NAME model1b COMMAND test_model1b)
  add_test(NAME moe_layer COMMAND test_moe_layer)
  add_test(NAME data_pack COMMAND test_data_pack)
endif()

# Cross-compile hints (CMake + Zig per AGENTS.MD 6):
#   x86_64 Linux:  cmake -S . -B build-x64 -DCMAKE_SYSTEM_PROCESSOR=x86_64
#   ARM64 macOS:   cmake -S . -B build-arm64 -DCMAKE_OSX_ARCHITECTURES=arm64
#   Windows:       cmake -S . -B build-win (MSVC or clang-cl)
#   Zig cc:        cmake -S . -B build-zig -DCMAKE_C_COMPILER="zig;cc"
