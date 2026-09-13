# JIMOTONO
どんな環境でも動作する、本当のローカル(地元の)LLMを目指すプロジェクト。

## 1. プロジェクト概要

**目的**：GPUなしで動作する、日本語・英語（コーディング）・ツールコールに特化したMoEベースLLMの推論・学習エンジンを、C言語（一部ASM）でゼロから構築する。

**設計目標**：
- RAM使用量：dense部分 2GB以下（INT4）、全体 8GB以下
- 計算負荷：アクティブパラメータ 100M、1トークンあたりのカーネル起動回数 50回以下
- 推論速度：N100で5 tok/s以上、Ryzen AI 360で30 tok/s以上
- 性能：Qwen3.8-27B相当（日本語・コーディング・ツールコール）
- モデル規模：総10〜20B / アクティブ100M

**クロスプラットフォーム**：
- CPU：x86_64（AVX2 / AVX-512）、ARM64（NEON）
- OS：Linux、macOS、Windows(どれもネイティブ、可能ならARM対応によってAndroidも)
- SIMD分岐：`#ifdef __AVX512__` / `#ifdef __AVX2__` / `#ifdef __ARM_NEON`
- エンディアン：リトルエンディアン前提。ビッグエンディアンは非対応
- ファイルI/O：io_uring（Linux）、kqueue（macOS）、IOCP（Windows）。抽象化レイヤで吸収

**設計思想**：
- 重みはHF形式やPyTorch形式である必要はない。独自バイナリ形式（ブロック量子化済み、CPUキャッシュラインに整列）
- OpenAI互換APIで応答できれば、内部実装は完全に自由
- OSページキャッシュをバイパス（O_DIRECT）し、SSDを直接操作
- 低ビット前提でアーキテクチャを最初から設計。T-MAC方式のルックアップテーブルを採用

## 2. アーキテクチャ

### 2.1 全体構成

| 要素 | 選択 | 参照元 |
|---|---|---|
| 線形注意 | Gated DeltaNet-2 | https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention |
| 線形：フル比率 | 4:1〜7:1（線形寄り） | https://aclanthology.org/2024.emnlp-main.916.pdf |
| MoE粒度 | 細粒度（1エキスパート2〜4M、数千個） | https://huggingface.co/FlameF0X/TinyMoE-100m-2x8 |
| 共有エキスパート | 常時オン2つ | https://deepseekv4.space/blog/deepseek-v4-technical-report |
| ルーティング予測 | StickyMoE soft-hard variant | https://arxiv.org/html/2607.08780v1 |
| KVキャッシュ | CSA（4:1圧縮 + top-1024選択） | https://deepseekv4.space/blog/deepseek-v4-technical-report |
| 語彙 | llm-jp-tokenizer v2.2（48,588語彙） | https://github.com/llm-jp/llm-jp-tokenizer |

### 2.2 混合精度設計

| 要素 | 精度 | 参照元 |
|---|---|---|
| 共有エキスパート | INT8 | 常時発火 |
| ルーティングエキスパート down_proj | INT4 | https://huggingface.co/OsaurusAI/ZAYA1-8B-JANGTQ_K |
| ルーティングエキスパート gate/up | INT2 | https://huggingface.co/OsaurusAI/ZAYA1-8B-JANGTQ_K |
| アテンション射影 | INT8 | 毎トークン発火 |
| 埋め込み/出力ヘッド | INT8（tied embeddings） | 重要かつ小さい |

FRI-MxMoE：プロファイリング不要でエキスパート内サブレイヤーの異なるロバスト性を捉え、最大15.7倍のプロファイリング高速化。
https://aclanthology.org/2026.acl-long.982/

## 3. 推論エンジン

### 3.1 プリフィル段階

- DeepSeek V4のCSA：Sliding window（128トークン）+ 学習済みKV圧縮（4:1）+ Heavy compression（128:1）
- FP4 KVキャッシュ：E2M1形式、16チャネルごとに1つのE4M3スケール

### 3.2 デコード段階

- 投機的エキスパート読み込み：NeuroPrefetcher方式。レイヤ0実行後、単一の予測器が全下流MLP層のスパース活性を予測し、必要な行のみを事前フェッチ
  https://github.com/nobeldhar/NeuroPrefetcher
- Delta Prefetching：活性ニューロンの82〜85%はトークン間で持続。残り15〜18%のみをio_uring + O_DIRECTで読み込む
- Router-lookahead prefetch：次層のルーティングは現層のpost-attention状態から71.6%予測可能

### 3.3 LUT（ルックアップテーブル）

- T-MAC方式：int1/2/3/4 × int8/fp16/fp32のmpGEMMを逆量子化なしで実行
  https://ar5iv.labs.arxiv.org/html/2407.00088
- 性能：BitNet-b1.58-3Bでシングルコア30 tok/s、8コア71 tok/s（M2-Ultra）、Raspberry Pi 5で11 tok/s
- エネルギー効率：llama.cpp比で60〜70%削減
- 学習時：HGQ-LUT方式。LUT-Dense層を通常のテンソル演算で学習し、推論時にLUTにコンパイル

### 3.4 分岐予測の活用

SpecBranch：CPUの分岐予測にインスパイアされた投機的デコーディングで1.8x〜4.5x高速化、ロールバックトークン50%削減。

## 4. 学習エンジン

### 4.1 メモリ戦略

| 手法 | 効果 | 参照元 |
|---|---|---|
| 勾配チェックポインティング | 活性化メモリをO(√n)に | — |
| 8bitオプティマイザ | オプティマイザ状態を1/3.8に | bitsandbytes-CPU |
| ES-MoE | エキスパートパラメータをCPU/SSDにオフロード | https://ina.kaist.ac.kr/projects/esmoe/ |
| SmartUpdate | オプティマイザ状態をSSD上で直接更新 | — |
| RAMディスク | 重み更新をRAM上で | 180GB RAMがある場合 |

### 4.2 カーネル融合とパイプライン

- SwiGLU融合：gate/up/downを1カーネルで処理
- RMSNorm + RoPE融合：メモリアクセスを最小化
- O_DIRECT + io_uring：SSDからのエキスパート読み込みを非同期化
- 投機的プリフェッチ和集合：S∪L∪Mの3予測器の和集合を先読み

### 4.3 RL手法

| 手法 | 参照モデル | 順伝播回数 | 参照元 |
|---|---|---|---|
| GRPO | 不要（KL削除） | 生成N回 + 更新1回 | https://github.com/alex000kim/nanoRL |
| ORPO | 不要 | 1回 | SFTと同時 |
| SimPO | 不要 | 1回 | 選好学習 |
| SPO | 不要 | 1回 | 過最適化抑制 |

RLVR：コード実行、ツールコール成否をルールベース報酬として使用。

## 5. データセット

| 目的 | データセット | 参照元 |
|---|---|---|
| 日本語基盤 | LLM-jp Corpus v4（日本語6,880億トークン、コード218Bトークン） | https://github.com/llm-jp/llm-jp-corpus-v4 |
| コーディング | SwallowCode-v2 | https://huggingface.co/datasets/tokyotech-llm/swallowcode-v2 |
| ツールコール | Japanese Function Calling Dataset | https://huggingface.co/datasets/Tonari-no-usagi/Japanese_Function_Calling_Dataset |

## 6. 技術スタック

| レイヤー | 言語 | 理由 |
|---|---|---|
| 推論エンジンコア | C + ASM（ホットパスのみ） | Curieが採用 https://github.com/Scottcjn/colibri |
| 学習エンジンコア | C | メモリ管理とI/Oを極限まで最適化 |
| オーケストレーション | Go または Rust | APIサーバー、リクエスト処理 |
| ビルドシステム | CMake + Zig | クロスコンパイル |

## 7. エージェントへの指示

### 7.1 コーディング規約

- Cコード：C11標準。`restrict`修飾子を積極的に使用。SIMD intrinsicsは`#ifdef`でガード
- メモリ管理：アリーナアロケータまたはプールアロケータを実装
- エラーハンドリング：`errno`ベース。`goto cleanup`パターン
- クロスプラットフォーム：Linux/macOS/Windowsで同一のソースがビルド可能であること

### 7.2 禁止事項

- 既存ライブラリへの依存：llama.cpp、vLLM、SGLang、PyTorchを直接リンクしない
- HF形式への固執：モデル重みは独自バイナリ形式で保存
- GPU前提の設計：すべてCPUのみで動作

## 8. 成功基準

| 指標 | 目標 |
|---|---|
| 推論速度（N100） | 5 tok/s以上 |
| 推論速度（Ryzen AI 360） | 30 tok/s以上 |
| RAM使用量（dense） | 2GB以下（INT4） |
| RAM使用量（全体） | 8GB以下 |
| カーネル起動回数 | 50回/token以下 |
| 1トークンあたりNVMe読み込み | 103 MiB以下 |
| ルーティング予測精度 | 99%以上 |
| スイッチ率削減 | 59%以上 |
| キャッシュミス削減 | 3.92倍以上 |
| 日本語MT-Bench | GPT-4o以上 |
| SWE-bench Pro | 50以上 |
| ツールコール成功率 | 90%以上 |