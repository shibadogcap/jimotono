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

if(JIMOTONO_NATIVE AND NOT MSVC)
  add_compile_options(-march=native)
endif()

# ---- core library ----
add_library(jimotono_core
  src/arena.c
  src/tmac.c
)
target_include_directories(jimotono_core PUBLIC include)
# tmac.c uses round()/isfinite() from <math.h>; libm needed on Linux.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(jimotono_core PUBLIC m)
endif()

# ---- microbenchmarks (ROADMAP 5.2: harness + cases + metric) ----
if(JIMOTONO_BUILD_BENCH)
  add_executable(bench_gdn2 bench/kernel/gdn2.c)
  target_link_libraries(bench_gdn2 PRIVATE jimotono_core)

  add_executable(bench_tmac bench/kernel/tmac_lut.c)
  target_link_libraries(bench_tmac PRIVATE jimotono_core)

  add_executable(test_arena bench/kernel/test_arena.c)
  target_link_libraries(test_arena PRIVATE jimotono_core)

endif()

enable_testing()
if(JIMOTONO_BUILD_BENCH)
  add_test(NAME arena COMMAND test_arena)
  add_test(NAME tmac COMMAND bench_tmac)
endif()

# Cross-compile hints (CMake + Zig per AGENTS.MD 6):
#   x86_64 Linux:  cmake -S . -B build-x64 -DCMAKE_SYSTEM_PROCESSOR=x86_64
#   ARM64 macOS:   cmake -S . -B build-arm64 -DCMAKE_OSX_ARCHITECTURES=arm64
#   Windows:       cmake -S . -B build-win (MSVC or clang-cl)
#   Zig cc:        cmake -S . -B build-zig -DCMAKE_C_COMPILER="zig;cc"
