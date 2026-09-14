# P3推論エンジン設計（P3b・推論のみ）

- branch: `docs/p3/inference`（本wtのみ。コミットしない）
- 性質：文書のみ。実装禁止、重負荷実行なし。
- 前提：`analysis/p3-architecture.md`（P3a）を最優先前提とし、矛盾させない。矛盾発見時は本書を直さず設計会議へ申送る（Phase Gの再拡大は禁止）。
- 整合対象（推論側のみ輸入・再定義しない）：`analysis/kv-cache-design.md`、`analysis/decoding-architecture.md`、`analysis/active-upper-bound.md`、`analysis/p3-design-inputs.md` §(a)〜§(c)、`analysis/roofline.md` D4・D6・D7、`analysis/training-speedup.md` P0/P1（I/O重ね合わせ・W8A8本線の依存としてのみ）。
- 非スコープ：P3c（学習エンジン本体）・実装・Phase G再拡大は禁止。語彙方針・RL・データセット等の4文書外への拡大はしない（推論のみ。必要箇所は依存として輸入する）。
- コスト表記：相対コストは Phase G Step 4 micro（`jt_gemm_mat_f32`、Mr=12×Nr=4）を「中」とした相対（小＜中＜大）。絶対人月・絶対期間は記さない。
- 数値の使い分け：帯域上限は公称値（N100 8B／Ryzen 2.6B）を設計目標に使わず、保守値（N100 約1.3B・Ryzen 約0.4B）を上限目安に使う（P3a §4と同一）。

## 0. 現状ギャップ（冒頭明記）

- **現状は学習最適化のみであり、推論パスは未整備である。** 既存資産（`train_longrun`、Phase G Step 4 micro GEMM、`analysis/roofline.md` D4–D7、`analysis/f2-breakdown.md`）はいずれも学習側（MoE fwd/bwd＋head＋optim8）の帯域律速・GEMM化・sync分析であり、推論の prefill／decode／KV永続／量子化推論パス／G1融合のいずれも実装・実測されていない。
- したがって本書は推論パスの**設計のみ**を固定する。受入はすべて「P3推論実装時の実測」をゲート条件とし、本書内で効果の上乗せ計上・速度値の断定は行わない（Phase G効果への上乗せ計上は禁止）。
- P3a固定事項の再掲（本書の境界条件・再定義ではない）：総10〜20B／アクティブ100M基準、線形:GDN-2・フル=4:1〜5:1、フル層KVのみ、MoE細粒度（2〜4M×数千・top-8〜16・共有2常時オン）、V=48,588＋factorized head・untie既定、dense≦2GB／全体≦8GB、起動≦50/token、NVMe≦103MiB/token、N100 5tok/s・Ryzen 30tok/s。

## 1. Prefill：CSA2＋FP4＋Bounded Replay（フル層のみ適用）

- **決定事項**：prefillのKV設計は `analysis/kv-cache-design.md` P3-1〜P3-4に準拠する。適用対象は**フル層のみ**（全層の約1/5。4:1時20%、5:1時約17%）。線形層（GDN-2）は固定サイズ再帰状態でKV不要のため対象外であり競合しない。内容：(i)層間共有（Reindex/Reuse相当の静的割当。キャッシュ共有とindex再利用は分離可能）、(ii)FP4格納（E2M1・16ch毎E4M3スケール・RoPE適用後量子化・注意時オンフライ逆量子化。事前逆量子化は不可）、(iii)SWA相当の局所KVはFP8維持＋SSD非永続＋Bounded Replay（中断再開時は末尾 n_win=128トークンのみ再計算。永続は圧縮済みFP4グローバルKVのみ）。CED本体・HCA復活・indexer別圧縮経路（V4式）は非採用。indexer Kはmain KVからの投影（CSA2式）に統一する。
- **根拠（参照URL）**：モデル https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash 、論文 https://www.alphaxiv.org/abs/2609.deepseek-v4-1-flash.pdf 、FP4実装構造参照 https://deepwiki.com/legend/deepseek-v4-vllm-patches/3-nvfp4-kv-cache-quantization ・ https://github.com/danielwoz/vllm-dspark-nvfp4 （フォーマット定数はV4.1側に従い、カーネル構造＝fused quant/store＋展開後演算再利用のみ借用。直接リンク禁止）、解説 https://dowithsudo.com/blog/deepseek-v4-1-flash-architecture/ 、V4 CSA前提 https://deepseekv4.space/blog/deepseek-v4-technical-report 。
- **未解決課題**：QATなしFP4 KVの精度未検証（層単位FP8フォールバックを必須付帯 P3-2として残す。感度評価は学習設計側と連携）。線形層を挟んだCSA層間共有の静的割当はP3本体設計時に確定（kv-cache §3(a)申送り）。DRAM一時プール（128×フル層SWA状態量×同時セッション数）のサイジング未算定。V4.1形式（per-16 E4M3・global scaleなし）とNVFP4（per-64 UE8M0）混同の防止が要る。
- **実装コスト（相対）**：中（FP4オンフライ逆量子化＝16エントリLUT×スケール乗算＋nibble pack隣接配置＋RoPE後量子化の走査順維持が主。層間共有の静的割当自体は小。QATなし精度検証を含めると中。大にはしない）。
- **依存**：P3a §1・§2（GDN-2・4:1〜5:1。フル層のみ対象の根拠）、`analysis/kv-cache-design.md` §2–§4・P3-1〜P3-4、`analysis/p3-design-inputs.md` §(c)C2（prefillはGEMM体制・計算律速）、`analysis/training-speedup.md` P0-2（`pread`隣接配置・O_DIRECT実測採用のI/O思想のみ）。
- **ゲート条件**：適用対象がフル層のみであること（線形層へのKV適用は設計改訂）。事前逆量子化の混入禁止（帯域利得を消すため）。スケールは同一キャッシュライン隣接配置（別ページのメタデータ読み禁止）。FP4導入は層単位FP8フォールバック付きであること。CED本体の混入禁止。

## 2. Decode：GDN-2＋ルーティング予測＋投機的先読み

- **決定事項**：decodeは `analysis/decoding-architecture.md` §4の採用決定に準拠する。(a)線形層はGDN-2逐次再帰形（`k3_kda_step`相当をGDN-2係数形で。チャンク形＝UT変換逆行列等は学習エンジン側に隔離しデコードCエンジンに混入させない）。KDAは独立採用せず縮退フォールバック（erase/write両ゲートを同一スカラに束縛するとKDAに還元）として同一コードパスに内包。工学的詳細としてshort-conv frontend（k=3〜4・SiLU融合）、q/k L2Norm、sigmoid有界減衰（g_min=-5、α∈(e^-5,1)）、full-rank出力ゲート、A_logのhead-index規約を借用。(b)ルーティング予測はStickyMoE soft-hard variantを採用（ルーティング熱の記録＋ホットpinへの接続は§5のNVMe受入条件として使う）。(c)投機的先読みは3予測器の和集合（S∪L∪M相当）を先読みする：NeuroPrefetcher方式（レイヤ0実行後、単一予測器が全下流MLP層のスパース活性を予測し必要行のみ事前フェッチ）＋Delta Prefetching（活性の82〜85%はトークン間持続。残り15〜18%のみ io_uring＋O_DIRECTで読む）＋Router-lookahead prefetch（次層ルーティングは現層post-attention状態から71.6%予測可能。batch-union＝同一expertは全positionで1回だけ読むと併用）。SGLangは構造参照のみ（融合順序・チャンク/再帰切替・conv1d更新パターン）。直接リンク禁止。
- **根拠（参照URL）**：GDN-2論文 https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention ・全文 https://arxiv.org/html/2605.22791 ・実装 https://github.com/NVlabs/GatedDeltaNet-2 、デコード手本 https://github.com/FareedKhan-dev/kimi-k3-in-c ・裏取り https://github.com/sehaxe/burn-kda 、StickyMoE https://arxiv.org/html/2607.08780v1 、NeuroPrefetcher https://github.com/nobeldhar/NeuroPrefetcher 、Router-lookahead実測・batch-union・PIPE/O_DIRECT思想 https://github.com/JustVugg/colibri （PILOT=71.6%、`docs/tuning.md`・`docs/benchmarks.md`）、SGLang構造参照 https://github.com/sgl-project/sglang/blob/4a50cd78/sgl-kernel/csrc/cpu/mamba/fla.cpp ・文書 https://docs.sglang.io/docs/hardware-platforms/cpu_server 。
- **未解決課題**：GDN-2学習カーネル（チャンクWY非対称erase因子＋gate-aware backward）のC学習エンジン側コスト未確定（P3c申送り。本書ではdecode逐次のみ扱う）。ルーティング予測精度99%以上・スイッチ率59%減・キャッシュミス3.92倍減（AGENTS.md §8）の実測未達。Delta持続率82〜85%・lookahead 71.6%は他系実測値であり自系実測で再取得要。MTP等の投機デコードは本書の対象外（P3本体の速度オプション。MTPはint8 head必須の制約付きのため§3と連動）。
- **実装コスト（相対）**：中（decode逐次再帰自体はKDA手本の係数差分で小だが、b/w 2系統＋3予測器の先読み機構＋batch-unionを含めると中。学習側チャンクWYは含まない）。
- **依存**：P3a §1・§3（GDN-2・細粒度MoE top-8〜16・共有2・capacity_factor=1.5・renormalize有・counting sort決定論的順序）、`analysis/decoding-architecture.md` §1–§4、`analysis/roofline.md` D4（GDN-2 decode AI 0.32・帯域律速）・D7 G1、`analysis/training-speedup.md` P0-1/P0-2（ルーティング熱pin・I/O重ね合わせ・batch-union・PILOT相当）、AGENTS.md §3.2・§8。
- **ゲート条件**：decode等価性はkimi-k3-in-c式3段階ゲート検証（teacher forcing／greedy／incremental）をGDN-2係数形で踏襲しbit同一契約（`-ffp-contract=off`相当）を維持すること。チャンク形をデコードCエンジンに混入させないこと。予測器の効果はpin-hit率＋Delta持続率で§5のNVMe 103MiB/token内に収める実測で示すこと（論文値の転載で代替不可）。

## 3. 量子化：INT4/INT2＋T-MAC LUT（W8A8本線との使い分け）

- **決定事項**：混合精度はP3a §6（AGENTS.md §2.2維持＋§5 untie既定）に準拠する。共有expert INT8・routed down INT4・routed gate/up INT2・注意射影INT8・emb/head INT8（headはfactorized 1024→256→48,588・emb full＋head factorizedのuntie）。使い分け：**W8A8 INT8経路を本線**（decode GEMVを含む確実な帯域半減＋整数SIMD/AMX加速。ブロック量子化＋AMX／AVX2-VNNI／NEON-I8MM分岐）、**T-MAC LUT（INT2/INT4 routed）はGEMM体制（Mが大きいprefill・学習）で効く副線**。decode GEMVではQLUT構築（act走査＋列挙）がamortize不可のため本線にしない（roofline D4末尾・D6判定）。HGQ-LUT方式（LUT-Dense層を通常テンソル演算で学習し推論時にLUTへコンパイル）の検証はP3本体の判断事項。FRI-MxMoEはエキスパート内サブレイヤーのロバスト性差の捉え方としての参照に限定する。
- **根拠（参照URL）**：混合精度参照 https://huggingface.co/OsaurusAI/ZAYA1-8B-JANGTQ_K 、T-MAC https://ar5iv.labs.arxiv.org/html/2407.00088 、W8A8 INT8本線 https://docs.sglang.io/docs/hardware-platforms/cpu_server 、FRI-MxMoE https://aclanthology.org/2026.acl-long.982/ 、容量勘定 `analysis/p3-design-inputs.md` §(b)B2・帯域上限 `analysis/active-upper-bound.md`。
- **未解決課題**：INT2 gate/up・INT4 downの精度実測未実施。HGQ-LUT学習→推論コンパイルの検証未実施。Q4_K_M相当のスケール込み換算（約0.56B/p）と素朴0.5B/pの使い分けは§5のbyte/token見積りで明示要。MTP採用時はint8 head必須の制約（Colibri issue #8相当）が§2の投機オプションに波及する。
- **実装コスト（相対）**：中（W8A8 INT8本線のブロック量子化＋SIMD分岐が主。T-MAC LUTのGEMM副線を含めると中。decode GEMVへのLUT拡張は非スコープでありコスト計上しない）。
- **依存**：P3a §4〜§6（A100M基準・factorized head約12.7MB・総10〜20B容量勘定）、`analysis/training-speedup.md` §5・P0-3（W8A8本線・85%帯域効率はGEMMパス目標）、`analysis/roofline.md` D4–D6（全カーネル帯域律速・GEMM化でAI 0.5→23・LUTはGEMM体制のみ有効）、`analysis/p3-design-inputs.md` §(b)・§(c)C2。
- **ゲート条件**：dense≦2GB／全体≦8GBの容量勘定を崩さないこと。T-MAC LUTをdecode GEMVの本線として計上しないこと（GEMM体制でのみ有効）。factorizedはheadのみ・k=256・emb full INT8常駐・既定untieを崩さないこと。AGENTS.md §2.2の「tied embeddings」表記はP3本体設計への申送り事項とし本wtでは表記変更しないこと。

## 4. G1融合：GDN-2 S 7→2（GEMM化後の帯域残渣として位置づけ）

- **決定事項**：GDN-2 decode内 S の7パス→2パス融合（scale＋erase＋write＋readout。`src/gdn2.c:314-428`相当）をdecode実装の受入条件とする。削減は S 5パス分＝5×64KB＝**320KB/層**、20線形層で**6.4MB/token級**（`analysis/roofline.md` D7 G1）。位置づけは**GEMM化後の帯域残渣**である。帯域律速を動かす主レバーはD6のトークンバッチ（GEMM化、AI 0.5→23の理論値）であり、G1はdecode時の支配項になり得る唯一の融合としてGEMM化の後に残る帯域項を削るものとする。現行longrunの活性融合（F1–F3合計で全体約1%級）とは混同しない。
- **根拠（参照URL）**：`analysis/roofline.md` D4（GDN-2 decode AI 0.32・S=64KBを7回なめる構造）・D7 G1（削減見積・HIGH判定・decode実装の受入条件化）・D6（GEMM化が主レバー）、SGLang融合順序の構造参照 https://github.com/sgl-project/sglang/blob/4a50cd78/sgl-kernel/csrc/cpu/mamba/fla.cpp （直接リンク禁止・順序のみ）。
- **未解決課題**：融合後の削減トラフィック（S 5パス分）の実測未実施。RMSNorm＋GEMM融合（G2）・GEMM＋Epilogue（G3）は本書では優先度付けのみ（MEDIUM）であり設計しない。AVX-512/NEON移植時の融合形の再調整は未実施（AVX2 Mr=12×Nr=4のみ確定の前提を維持）。
- **実装コスト（相対）**：小〜中（G1単体の融合は小。decode受入条件としての実測裏付けまで含めると中。大にはしない。Phase G Step 4 micro＝中との混同禁止）。
- **依存**：P3a §1（G1はdecode実装の受入条件）、`analysis/roofline.md` D4・D6・D7、`analysis/decoding-architecture.md` §1–§3（逐次再帰形の維持が融合の前提）、Phase G Step 4 micro（GEMMパス。GDN-2融合と混ぜない）。
- **ゲート条件**：G1融合はdecode実装時に削減トラフィック（S 5パス分＝320KB/層）を実測で示すこと。GEMM化（トークンバッチ）の効果に上乗せ計上しないこと（非スコープ混入の確認）。チャンク形を融合口実にデコードCエンジンへ混入させないこと。

## 5. 推論スループット目標：N100 5 tok/s・Ryzen 30 tok/sとの対応（保守値準拠）

- **決定事項**：目標はAGENTS.md §8の N100 5tok/s・Ryzen AI 360 30tok/sを維持し、対応付けは `analysis/active-upper-bound.md` §3–§4の保守値準拠で行う。公称上限（η=0.5・KV微小・オーバーヘッド無視の理論上限：N100約8B／Ryzen約2.6B）は設計目標・受入条件に使わない。**上限目安には保守値（η=0.1＋KV/スケール/固定費で×0.8）：N100約1.3B・Ryzen約0.4Bを使う**。P3想定 byte/token≒63〜100MB（100M INT4≒50MB＋縮小head INT8約12.7MB＋KV微小）に対し、必要eff_BWはN100 5×100MB=0.5GB/s（peak比約1.3%）・Ryzen 30×100MB=3.0GB/s（peak比約3.8%）であり、100M基準は保守値に対してもN100約13×・Ryzen約4×のマージンがある。よって帯域は律速ではなく、律速候補は未縮小head（§3で除去済み）・起動固定費（≦50/token）・NVMe IOPS/遅延（≦103MiB/token）・スレッドsync逓増とする。NVMe予算（想定約100MB/token）は境界上のため、pin-hit率（§2のStickyMoE熱pin相当）＋Delta持続率で103MiB内に収めることをP3受入条件とする。η=0.3中間点（N100約3.8B・Ryzen約1.3B）はA300-500M選択肢の可否判断の参照点に限定し、本書で既定化しない（P3a §4の条件(i)–(iv)付き選択肢を維持）。
- **根拠（参照URL）**：帯域式 `tok/s≒eff_BW/bytes_per_token`（`analysis/p3-design-inputs.md` A1、`analysis/active-upper-bound.md` §1と同一）、較正はLFM2-24B-A2B（Q4_K_M約1.3GB/tokenで112tok/s→実効約145GB/s、η≒0.57。https://huggingface.co/LiquidAI/LFM2-24B-A2B 、https://www.liquid.ai/blog/lfm2-24b-a2b ）を存在証明に限定。toks/s cross-vocab直接比較は行わない。技術報告 https://arxiv.org/html/2511.23404 、配置実測の参照 https://github.com/JustVugg/colibri 。
- **未解決課題**：η実測（STREAM未計測±30%）、単ch・小GEMV・競合時の下振れ、長文脈・大バッチ時のKV加算、O_DIRECT/SSD待ち混入、スレッドsync逓増（6T 6.2%→12T 11.3%。`analysis/f2-breakdown.md`）。TTFT実数（1k tokens数秒級の設計値）はP3実測で確定（本書は式と方針のみ）。prefill目標は計算律速（AMX/AVX2）のため本帯域式の対象外（C2）。
- **実装コスト（相対）**：小（目標値自体のコストはない。コストは§1–§4のKV微小化・I/O重ね合わせ・head縮小・G1融合側に計上）。
- **依存**：P3a §4（A100M基準＋保守表・A300-500M条件付き選択肢）、`analysis/active-upper-bound.md` §2–§4、`analysis/p3-design-inputs.md` §(a)・§(c)C1–C2、`analysis/vocab-budget.md`（head縮小）、`analysis/kv-cache-design.md`（KV微小）、`analysis/f2-breakdown.md`（sync実測）、AGENTS.md §8。
- **ゲート条件**：公称8B/2.6Bを受入条件・設計目標として引用しないこと。LFM2/Gemma値を同語彙速度比較に使わないこと。NVMe 103MiB/token内にpin-hit率＋Delta持続率で収める実測をP3受入条件に含めること。A300-500M選択肢の採用はP3a §4の(i)–(iv)＋η実測を満たした設計改訂でのみ行うこと。

## 参照URL一覧（本書の推論範囲に限定・P3aとの重複は維持）

- https://www.alphaxiv.org/abs/2609.deepseek-v4-1-flash.pdf
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- https://deepwiki.com/legend/deepseek-v4-vllm-patches/3-nvfp4-kv-cache-quantization
- https://github.com/danielwoz/vllm-dspark-nvfp4
- https://dowithsudo.com/blog/deepseek-v4-1-flash-architecture/
- https://deepseekv4.space/blog/deepseek-v4-technical-report
- https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention
- https://arxiv.org/html/2605.22791
- https://github.com/NVlabs/GatedDeltaNet-2
- https://github.com/FareedKhan-dev/kimi-k3-in-c
- https://github.com/sehaxe/burn-kda
- https://arxiv.org/html/2607.08780v1
- https://github.com/nobeldhar/NeuroPrefetcher
- https://github.com/JustVugg/colibri
- https://aclanthology.org/2024.emnlp-main.916.pdf
- https://huggingface.co/FlameF0X/TinyMoE-100m-2x8
- https://aclanthology.org/2026.acl-long.982/
- https://huggingface.co/OsaurusAI/ZAYA1-8B-JANGTQ_K
- https://ar5iv.labs.arxiv.org/html/2407.00088
- https://docs.sglang.io/docs/hardware-platforms/cpu_server
- https://github.com/sgl-project/sglang/blob/4a50cd78/sgl-kernel/csrc/cpu/mamba/fla.cpp
- https://huggingface.co/LiquidAI/LFM2-24B-A2B
- https://www.liquid.ai/blog/lfm2-24b-a2b
- https://arxiv.org/html/2511.23404
- https://github.com/llm-jp/llm-jp-tokenizer
