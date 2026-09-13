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
  src/optim8.c
  src/esmoe.c
)
target_include_directories(jimotono_core PUBLIC include)
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

  add_executable(test_optim_offload bench/kernel/test_optim_offload.c)
  target_link_libraries(test_optim_offload PRIVATE jimotono_core)

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
  add_test(NAME optim_offload COMMAND test_optim_offload)
endif()

# Cross-compile hints (CMake + Zig per AGENTS.MD 6):
#   x86_64 Linux:  cmake -S . -B build-x64 -DCMAKE_SYSTEM_PROCESSOR=x86_64
#   ARM64 macOS:   cmake -S . -B build-arm64 -DCMAKE_OSX_ARCHITECTURES=arm64
#   Windows:       cmake -S . -B build-win (MSVC or clang-cl)
#   Zig cc:        cmake -S . -B build-zig -DCMAKE_C_COMPILER="zig;cc"
