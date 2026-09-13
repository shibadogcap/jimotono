#!/bin/sh
# scripts/bench_env.sh: ベンチ環境情報の収集 (ROADMAP 5.1 外乱記録用)。
# 使い方: sh scripts/bench_env.sh | tee results/bench/<date>-env.txt
set -eu
echo "=== date ==="; date
echo "=== git ==="; git rev-parse HEAD; git status -sb | head -n 5
echo "=== cpu ==="; uname -m
sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 "model name" /proc/cpuinfo 2>/dev/null || true
echo "=== cc/cmake ==="; cc --version | head -n 2; cmake --version | head -n 1
echo "=== load ==="; uptime
