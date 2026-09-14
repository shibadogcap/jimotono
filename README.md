# JIMOTONO

GPUなしで動作する、日本語・英語（コーディング）・ツールコール特化のMoEベースLLMの
推論・学習エンジンをC11で構築するプロジェクト。詳細は `AGENTS.MD` を参照。

- MoE: 総10-20B / active 100M既定、細粒度expert + 共有expert 2つ、StickyMoE routing
- Attention: Gated DeltaNet-2 (linear:full = 4:1-5:1)、CSA KV圧縮 + FP4 cache
- 精度: 共有expert/attention INT8、routed down INT4 / gate-up INT2、emb/head INT8 (factorized)
- I/O: `O_DIRECT` + `io_uring` (Linux) / `F_NOCACHE` + `kqueue` (macOS) /
  `FILE_FLAG_NO_BUFFERING` + IOCP (Windows、将来) を抽象化層で吸収
- 語彙: llm-jp-tokenizer v2.2 (48,588語彙、Apache-2.0)
- 目標: N100 5 tok/s以上、dense 2GB以下 / 全体8GB以下、NVMe 103MiB/token以下

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

3OS同一ソース (Linux/macOS/Windows) が必須。SIMDはTU別opt-in
(`moe_gemm.c`/`moe_layer.c`/`train_bwd.c`/`w8a8.c`に`-mavx2`のみ、AVX-512なし)。

## Test

```sh
ctest --test-dir build --output-on-failure
```

短時間correctnessのみ。重負荷bench (`train_proxy100m`等) はctest外。

## CI

`ci` workflow (`.github/workflows/ci.yml`): cmake configure (Debug) -> build -> ctestをmatrix化。

- 既存4: linux-gcc / linux-clang / macos-clang / windows-msvc
- 追加4: linux-arm64-gcc (ubuntu-24.04-arm) / macos14-arm64-clang / macos13-x64-clang / windows-arm64-msvc
- 命令セット検証はCI不可: コンパイル分岐のみ担保し、実行時dispatchはログに留める。

## License

Apache-2.0 (`LICENSE`)。llm-jp-tokenizer語彙のApache-2.0と整合。
