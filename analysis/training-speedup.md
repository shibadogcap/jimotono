# 調査項目3：学習高速化 — Colibri/Curie＋外部手法の適用選定（P3設計入力）

- branch: `docs/p3/survey-train-design`（本wtのみ。コミットしない）
- 性質：調査＋文書のみ。実装禁止、重負荷実行なし（Web調査＋既存計測値の再利用のみ）。
- 結論先行：Colibri系の「学習済みルーティング熱によるホットピン留め＋O_DIRECT/io_uring重ね合わせ＋バッチ和集合」は
  そのままP3のES-MoE/プリフェッチ設計に採用可能（P0）。SGLang式W8A8 INT8 MoE（帯域半減＋AMX）は混合精度表と整合しP0。
  Megatron細粒度offloadは粒度ポリシー（安い層は再計算／重い層は退避）のみ輸入（P0・概念）。
  MEFT疎アダプタ更新とHyGINサイズ別経路はP1（SSD直接更新・fetch経路の拡張として）。

## 0. 名称整理（重要）

- AGENTS.md §6は推論エンジンコアの参考として `https://github.com/Scottcjn/colibri`（Curie）を挙げるが、
  現在の正準リポジトリは **`https://github.com/JustVugg/colibri`**（約30k stars、Apache-2.0、単一Cファイル＋小ヘッダ構成）。
  Scottcjn名義は旧称・フォーク系列とみられる。本書はJustVugg版を正とする。
- Colibriは**推論エンジン**であり学習器ではない。「学習高速化」への適用は、
  Colibriの配置・I/O・キャッシュ機構をP3学習エンジン（ES-MoE＋SmartUpdate＋ckpt）の設計に転用する形に限定する。
  語彙スパース化の実装には着手しない（P4記録のみ。タスク禁止事項）。

## 1. Colibri/Curie 精読（`github.com/JustVugg/colibri`）

参照：README・`docs/tuning.md`・`docs/benchmarks.md`・`docs/cuda.md`（2026-09-14閲覧）。

### 1.1 アーキテクチャ要点（実測値付き）

- **3階層配置（VRAM / RAM / NVMe）を単一メモリ階層として扱う**。容量不足は速度にのみ影響し、意味（精度・ルータ決定）を変えない。
- GLM-5.2（744B total / 40B active）の内訳：dense部（attention＋共有expert＋埋め込み ≈17B params）をint4常駐（約9.9GB）。
  19,456 routed experts（75 MoE層×256＋MTPヘッド、int4で約19MB/個、計約370GB）はディスク常駐＋オンデマンド転送。
- **学習キャッシュ（JIT-for-weights）**：使用実績のルーティング熱をモデル脇の `.coli_usage` に毎ターン追記し、
  起動時にホットexpertを自動pin（`PIN=auto`、`PIN_GB`、適応LRUと予算共有、自動pinはLRU容量を侵さない上限付き）。
  `--repin N` で安全なターン境界でpin集合を入替（減衰ヒートマップ、25%ヒステリシス、4スワップ上限でスラッシング防止）。
  配布コンテナに `.coli_usage` 同梱例あり（6,592,176 selection記録でpin-hit率 56.9%→72.8%、約16pt改善）。
  部分ミラー計画（`coli mirror plan/stage/verify`）も `.coli_usage` の熱順でシャード優先度付け。
- **I/Oはエンジンの一部**：expert 3行列を隣接配置し1 `pread` で読む。非同期I/Oプール `PIPE=1`（既定ON、matmulとpread重ね合わせ、
  ディスクサービス約−18%）。`O_DIRECT`（`DIRECT=1`）はDRAMキャッシュ付き高速NVMeで約＋34〜65%（Strix Haloで＋65%、GB10で4.25→9.69GB/s）
  だが、QLC/DRAM-less/仮想化ディスクでは中立〜負。**ドライブ依存＝実測採用**が公式方針。Linux限定バッチI/O `URING=1`（`PIPE=1`含意）。
- **Router-lookahead prefetch（`PILOT=1`）**：次層ルーティングは現層post-attention状態から**71.6%予測可能**と実測。
  バッチ和集合（batch-union：同一expertは全positionで1回だけ読む）と併用。
- **Dual-SSD**：モデル複製を2台に置き、帯域加重ハッシュで振分け。合計帯域＝2台の和（9＋3GB/s例で高速単独比約33%高速）。
  ミラーは検証のみ・**書込不可**（`.coli_usage`・`.coli_kv`等sidecarはprimaryのみ）。読込失敗はprimaryへ縮退。
- **MLA KV 57×圧縮**：32,768→576 floats/token。`.coli_kv` で会話またぎ永続化（再prefillゼロ、バイト同一）。
- **投機（MTP）**：2.2〜2.8 tokens/forward。ただしMTPヘッドは**int8必須**（int4でacceptance 0〜4%に崩壊、issue #8）。
  draftとverifyは同一カーネル系に固定（`SPEC_PIN=1`、issue #163）。Grammar-forced draftは制約JSONで約無料。
- **測定フロア**：6×RTX 5090全常駐 5.8〜6.8 tok/s、128GB CPU-only 約1.8 tok/s（warm）、25GB転送 streaming 0.05〜0.1 tok/s（cold）。
  速度はディスクで決まる設計。

### 1.2 「同時設計・SSD直接更新・ホットピン留め・744B/25GB」の対応付け

| タスク文言 | Colibri側の実体 | P3への写像 |
|---|---|---|
| 同時設計（co-design） | 形式（int4-gs64 routed／int8 MTP／bf16 denseを読込時量子化）＋配置プランナ（`coli plan/doctor/tune`）を一体開発 | 独自バイナリ形式（AGENTS §1設計思想）をP3で同時設計。量子化粒度・隣接配置・ヘッダ検証を形式仕様に含める |
| SSD直接更新 | エンジン自体は重み不変。更新されるのはsidecar（`.coli_usage`毎ターン、`.coli_kv`）でprimaryのみ書込・ミラー不変・O_DIRECT読込 | SmartUpdateのSSD上optimizer直接更新と同型：**更新系はprimary側に集約、読込専用複製には書かない**。O_DIRECTは実測採用 |
| `.coli_usage`ホットピン留め | 上記1.1の学習キャッシュ | StickyMoEのルーティング熱＋ES-MoE pinに直輸入（P0）。`mirror plan`相当の熱順ステージングも学習データ配置に応用可 |
| 744B MoE/25GB RAM | dense常駐＋expert転送で成立（ただし0.05〜0.1 tok/s） | 総10〜20B/全体8GB（AGENTS §8）への縮小版として成立。速度目標は別途§(c)で見積り |

## 2. Megatron-LM 細粒度 activation offloading（PR #1913系）

- 参照：`https://github.com/NVIDIA/Megatron-LM`（`docs/user-guide/features/fine_grained_activation_offloading.md`、
  `megatron/core/transformer/transformer_config.py` の `fine_grained_activation_offloading`／`offload-modules`／
  `activation-offload-fraction`）。RedNote協業、Megatron-Coreに統合。タスク指定のPR #1913は本機能の導入系譜として扱う。
- 内容：層単位ではなく**サブモジュール粒度**（`attn_norm`・`qkv_linear`・`core_attn`・`attn_proj`・`mlp_norm`・
  `expert_fc1`・`moe_act`・`fused_group_mlp`）でactivationを非同期にCPUへ退避し、計算と重ねる。
  細粒度recomputeと併用し、層のほぼ全activationをデバイス外へ出せる。PP各種・FP8・MTP・混成dense/MoE・A2A重ね合わせ対応。
  CUDAグラフ内にoffload対象を含められない暫定制限あり。
- 公式ポリシー：**安いモジュール（layernorm・moe_act）はrecompute、重いモジュール（core_attn・expert_fc1等）はoffload**。
- JIMOTONO（CPU-only）への輸入は**概念のみ**：退避先がRAM↔SSDになるだけで粒度ポリシーは同型。
  既存資産との対応：`analysis/f2-breakdown.md` でckpt 5.7ms/step・esmoe_io 0.7〜0.9ms/step（28M常駐域では律速でない）と測定済み。
  よって本手法は「総10〜20Bで常駐不能になったときの粒度ポリシー」としてP3仕様に織り込む（offload-fraction相当ノブの予約）。

## 3. MEFT（Memory-Efficient Fine-Tuning through Sparse Adapter）

- 参照：Hao et al., ACL 2024（`https://aclanthology.org/2024.acl-long.129`、`arXiv:2406.04984`、
  `https://github.com/CURRENTF/MEFT`）。
- 内容：大容量アダプタをCPUに置き、入力ごとに**疎活性上位ニューロンのみCPU→GPUへ取出す**。
  MoE的Key-Experts機構でCPU計算とPCIe転送を削減。GPU 48G→24G（約50%削減）で同等精度、同一予算ではPA/LoRAを上回る
  （NQ・SQuAD・Tool等の知識集約タスク）。
- JIMOTONOへの輸入：我々はGPUを持たないので方向を反転し、**「更新対象を疎化する」設計として採用**：
  routed-expertのうち発火分のみoptimizer step（SSD上直接更新＝SmartUpdateの疎化版）、Key-Experts相当のpail分割で
  RAM↔SSD転送を削減。日本語・ツールコール等の知識集約適応と相性がよい。P1（SmartUpdate拡張としてP3仕様化、実装は別phase）。

## 4. HyGIN（Hybrid CPU/GPU-Initiated Communication for MoE Training）

- 参照：Huang & Kim, IEEE Computer Architecture Letters vol.25 (Jul.-Dec. 2026), DOI `10.1109/LCA.2026.3711652`
 （`https://www.computer.org/csdl/journal/ca/2026/02/11601081/2hZ80QAVT8Y`）。
- 内容：MoE学習の集合通信を**メッセージサイズ別にCPU発行／GPU発行へ振分ける単一永続カーネル内ハイブリッド**。
  小メッセージはGPU発行が速く、大メッセージはCPU発行が速いという実測に基づく。
  Qwen3-235B・DeepSeek-V2-236Bのend-to-end模擬で学習時間−11%（対CPU発行）／−8%（対GPU発行）。
- JIMOTONOへの輸入：GPUがないため**fetch経路の切替則として転用**：小expert行は計算スレッド同期`pread`、
  大和集合は`io_uring`バッチ、というサイズ別二経路。効果はHyGIN論文値の引用ではなくP3で実測する。P1。

## 5. SGLang Xeon6 INT8 MoE（85%帯域効率）

- 参照：SGLang CPUサーバ文書（`https://docs.sglang.io/docs/hardware-platforms/cpu_server`：
  DeepSeek-V3.1-Terminus W8A8_INT8をXeon 6980Pで `--device cpu --quantization w8a8_int8 --enable-torch-compile --tp 6` 実行例）、
  Arm W8A8対応報告（PR #16045、dense＋MoEのINT8 GEMMをアーキ別ディレクトリ構成で共存）、
  Xeon 6（12ch DDR5/MRDIMM 8800MT/s、AMXでINT8 2048 ops/cycle/core）。
- 内容：**W8A8（重み8bit＋活性8bit）で帯域半減＋整数SIMD/AMX加速**。INT8 MoEカーネルをCPUパスに統合。
  「85%帯域効率」はサーバ級Xeon＋AMX＋ブロック量子化＋torch.compile融合で到達しうる実効帯域利用率の設計目標として扱う
  （JIMOTONOのN100/Ryzen feststellung値ではなく、GEMMパスの到達目標）。
- JIMOTONOへの輸入：AGENTS §2.2混合精度表（共有expert INT8・注意射影INT8・down INT4/gate-up INT2）と整合。
  P3のGEMMパス目標として **W8A8 INT8 MoE経路（ブロック量子化＋AMX／AVX2-VNNI／NEON-I8MM分岐）をP0採用**。
  T-MAC LUT（INT2/INT4 routed）とは使い分け：INT8＝確実な帯域半減の本線、LUT＝ `analysis/roofline.md` D4/D6の判定通り
  GEMM体制（Mが大きいprefill・学習）で効く副線。

## 6. 優先度付き選定（P3設計への組込み順）

| 優先度 | 手法 | 組込み内容 | 根拠・対応資産 |
|---|---|---|---|
| **P0-1** | Colibri学習キャッシュ輸入 | ルーティング熱記録（`.coli_usage`相当）＋ホットexpertのRAM pin＋`--repin`相当の減衰入替 | StickyMoE熱をES-MoE pinに直結。`f2-breakdown.md`のesmoe常駐測定と整合 |
| **P0-2** | Colibri I/O重ね合わせ | `pread`隣接配置＋async pool（PIPE相当）＋`O_DIRECT`実測採用＋Linux `io_uring`＋batch-union＋PILOT相当lookahead（71.6%前提で実測再取得） | デコード・学習fetch共通。AGENTS §3.2のNeuroPrefetcher/Delta/Router-lookaheadと同型 |
| **P0-3** | SGLang式W8A8 INT8 MoE | INT8経路を本線（ブロック量子化、AMX/AVX2-VNNI/NEON分岐）。帯域半減をP3見積りの既定係数に | §2.2混合精度表と一致。roofline D4「全カーネル帯域律速」の直接対策 |
| **P0-4** | Megatron細粒度ポリシー（概念） | 安価層recompute／重量層退避の粒度表＋offload-fraction相当ノブ予約 | 28M域では不要（f2実測）のため、10B超の常駐不能域への備えとして仕様化のみ |
| **P1-1** | MEFT疎更新 | 発火expertのみSSD上optimizer step（SmartUpdate疎化）、Key-Experts相当の分割転送 | 学習I/O削減。知識集約タスク適応と両立 |
| **P1-2** | HyGIN式二経路fetch | 小→同期pread／大→io_uringバッチのサイズ別切替 | 効果はP3実測で確定（論文値の転載禁止） |
| **P2** | Dual-SSD・NUMA・MTP | 複製帯域加算、メモリコントローラ分散、投機デコード | 速度オプション。MTPはint8 head必須の制約付き（Colibri issue #8） |

## 7. 非採用・先送り

- ColibriのGPU/VRAM tier・CUDA/Metal/Vulkan backend：GPU前提のため非採用（AGENTS §7.2準拠）。
- 語彙スパース化の実装：P4課題として記録のみ（タスク禁止事項）。
- toks/sのcross-vocab比較：禁止のため行わない。比較は帯域（GB/s）とbyte/tokenに正規化して行う（`p3-design-inputs.md` §(a)参照）。

## 8. 参照URL一覧

- `https://github.com/JustVugg/colibri`
- `https://github.com/JustVugg/colibri/blob/main/docs/tuning.md`
- `https://github.com/JustVugg/colibri/blob/main/docs/benchmarks.md`
- `https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp`
- `https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/fine_grained_activation_offloading.md`
- `https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/fine_grained_activation_offloading.html`
- `https://aclanthology.org/2024.acl-long.129`
- `https://arxiv.org/abs/2406.04984`
- `https://github.com/CURRENTF/MEFT`
- `https://www.computer.org/csdl/journal/ca/2026/02/11601081/2hZ80QAVT8Y`
- `https://docs.sglang.io/docs/hardware-platforms/cpu_server`
- `https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bringing-sglang-high-performance-llm-inference-to-arm-neoverse`
- 既存資産：`analysis/roofline.md`、`analysis/f2-breakdown.md`
