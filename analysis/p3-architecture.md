# P3アーキテクチャ（P3a最優先・他すべての前提）

- branch: `docs/p3/architecture`（本wtのみ。コミットしない）
- 性質：文書のみ。実装禁止、重負荷実行なし。本書はP3本体設計の前提固定であり、P3b/P3cには着手しない。
- 位置づけ：**本書が最優先であり、他すべてのP3文書の前提**。既存4文書＋pre-G文書と矛盾させない。矛盾発見時は本書を直さず設計会議へ申送る（Phase Gの再拡大は禁止）。
- 整合対象：`analysis/p3-design-inputs.md`、`analysis/decoding-architecture.md`、`analysis/kv-cache-design.md`、`analysis/training-speedup.md`、`analysis/active-upper-bound.md`、`analysis/vocab-budget.md`、`analysis/roofline.md`、`analysis/f2-breakdown.md`、`analysis/tokenizer-gap.md`、`analysis/phase-g-summary.md`（Phase G総括。本書はGの効果を上乗せ計上しない）。
- コスト表記：相対コストは Phase G Step 4 micro（`jt_gemm_mat_f32`、Mr=12×Nr=4）を「中」とした相対（小＜中＜大）。絶対人月・絶対期間は記さない。
- 数値の使い分け：帯域上限は公称値（N100 8B／Ryzen 2.6B）を設計目標に使わず、保守値（N100 約1.3B・Ryzen 約0.4B）を上限目安に使う（`analysis/active-upper-bound.md` §3–§4の指示を維持）。

## 0. 全体像（固定）

- 総10〜20B／アクティブ100M基準（§4・§6）。dense常駐2GB以下（INT4）／全体8GB以下、カーネル起動≦50/token、NVMe読込≦103MiB/token、N100 5tok/s・Ryzen AI 360 30tok/s（AGENTS.md §8）は変更しない。
- 線形注意GDN-2（§1）を比率4:1〜5:1（§2）で配置し、フル層KVのみCSA2＋FP4（`analysis/kv-cache-design.md` P3-1）を適用する。線形層は固定サイズ再帰状態でKV不要のため競合しない。
- MoEは細粒度・共有2・top-k（§3）、語彙V=48,588＋factorized head・untie既定（§5）。
- 学習・推論のI/O重ね合わせ・INT8本線・粒度ポリシーは `analysis/training-speedup.md` P0/P1を依存として輸入する（本書で再定義しない）。

## 1. 線形注意：GDN-2採用（KDAは参照・縮退フォールバック）

- **決定事項**：線形層はGated DeltaNet-2を維持（変更なし）。KDAは独立採用せず、**縮退フォールバック（erase/write両ゲートを同一スカラに束縛するとKDAに還元）として同一コードパスに内包**する。デコードC実装は逐次再帰形（`k3_kda_step`相当をGDN-2係数形で）。チャンク形（UT変換逆行列等）は学習エンジン側に隔離する。工学的詳細としてshort-conv frontend（k=3〜4、SiLU融合）、q/k L2Norm、sigmoid有界減衰（g_min=-5、α∈(e^-5,1)）、full-rank出力ゲート、A_logのhead-index規約を借用する。
- **根拠（参照URL）**：GDN-2論文 https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention 、全文 https://arxiv.org/html/2605.22791 、実装 https://github.com/NVlabs/GatedDeltaNet-2 。KDA参照 https://github.com/FareedKhan-dev/kimi-k3-in-c 、裏取り https://github.com/sehaxe/burn-kda 。SGLang構造参照 https://github.com/sgl-project/sglang/blob/4a50cd78/sgl-kernel/csrc/cpu/mamba/fla.cpp （融合順序・チャンク/再帰切替・conv1d更新パターンのみ。直接リンク禁止）。`analysis/decoding-architecture.md` §4の採用決定と同一。
- **未解決課題**：GDN-2学習カーネル（チャンクWY非対称erase因子＋gate-aware backward）のC学習エンジン側コスト未確定。デコード内Sの7パス→2パス融合（G1）はdecode実装の受入条件だが未設計（`analysis/roofline.md` D7 G1：20線形層で6.4MB/token級の支配項になり得る）。QATなしFP4 KVの精度は未検証（層単位FP8フォールバックを残す。`analysis/kv-cache-design.md` P3-2）。
- **実装コスト（相対）**：中（デコード逐次再帰はKDA手本の係数差分で小だが、学習側チャンクWY＋G1融合を含めると中。大にはしない）。
- **依存**：`analysis/decoding-architecture.md` §1–§4、`analysis/roofline.md` D4（GDN-2 decode AI 0.32・帯域律速）・D7 G1、`analysis/kv-cache-design.md`（フル層のみKV。線形層は対象外）、Phase G Step 4 micro（GEMMパス。GDN-2融合と混ぜない）。
- **ゲート条件**：デコード逐次再帰の等価性はkimi-k3-in-c式3段階ゲート検証（teacher forcing／greedy／incremental）をGDN-2係数形で踏襲し、bit同一契約（`-ffp-contract=off`相当）を維持すること。チャンク形をデコードCエンジンに混入させないこと。G1融合はdecode実装時に削減トラフィック（S 5パス分＝320KB/層）を実測で示すこと。

## 2. 線形：フル比率 4:1〜5:1（線形寄り）

- **決定事項**：線形（GDN-2）：フル（通常注意＋KV）を **4:1〜5:1**で維持する（4:1時フル約20%、5:1時約17%）。Kimi（≈2.9:1）・LFM2（3:1）より線形寄りだが、GDN-2の検索保持力（RULER multi-keyで顕著、erase側が主寄与）を根拠に正当化する。フル層側はCSA2＋FP4圧縮で補う。
- **根拠（参照URL）**：比率の参照元 https://aclanthology.org/2024.emnlp-main.916.pdf 。GDN-2品質根拠は§1と同一（1.3B/100BでMamba-2・GDN・KDA・Mamba-3中超える総合最良）。対照点：Kimi K3（69 KDA:24 MLA≈2.9:1）、LFM2-24B-A2B（30 conv:10 attn=1:3、https://huggingface.co/LiquidAI/LFM2-24B-A2B 、https://www.liquid.ai/blog/lfm2-24b-a2b 、https://arxiv.org/html/2511.23404 ）。「conv:attn=3:1でCPU実用速度が出る」事実は我々の4:1〜5:1が現実的範囲内であることの傍証に限定し、速度値の横並び比較はしない（`analysis/decoding-architecture.md` §2–§4、`analysis/p3-design-inputs.md` §(a)方法論と同一）。
- **未解決課題**：5:1端での長文脈検索劣化の定量未確定（RULER相当の自前評価はP3b以降）。線形層を挟んだCSA層間共有（Reindex/Reuse）の静的割当はP3設計時に確定する（`analysis/kv-cache-design.md` §3(a)申送り）。
- **実装コスト（相対）**：小（比率自体は静的割当であり、カーネル新規実装を生まない。コストはフル層圧縮側に計上）。
- **依存**：§1（GDN-2）、`analysis/kv-cache-design.md` §2–§4（フル層のみに乗算的に効く）、`analysis/p3-design-inputs.md` §(c)C2（prefillはGEMM体制・計算律速。T-MAC LUTはGEMM体制でのみ有効）。
- **ゲート条件**：比率変更は設計改訂扱いとする（場当たり調整禁止）。フル層削減技術（層間共有＋FP4＋Bounded Replay n_win=128）の適用対象を全層の約1/5に限定した byte/token 見積りをP3受入条件に含めること。

## 3. MoE粒度（細粒度・expert数・top-k）

- **決定事項**：細粒度MoEを維持する。**1エキスパート2〜4M、routed数千個、top-8〜16、共有エキスパート常時オン2つ**。総10〜20B÷2〜4Mよりrouted数は約2500〜10000の範囲に入り、top-k 8〜16時のrouted activeは約16〜64M＋共有＋注意射影＋縮小headで§4の100M基準に収まる。capacity_factor=1.5・renormalize有・aux μ=0.01はPhase G仕様（`analysis/phase-g-summary.md` §5）を引き継ぐ。ソートはcounting sort（E小規模）＋安定＋決定論的（`(expert_id, token_pos, slot)`辞書式完全順序、perm構築は単一スレッド）を維持し、将来数千規模ではradix切替え可能な構造とする。
- **根拠（参照URL）**：細粒度 https://huggingface.co/FlameF0X/TinyMoE-100m-2x8 、共有expert・CSAは https://deepseekv4.space/blog/deepseek-v4-technical-report 、ルーティング予測 StickyMoE soft-hard variant https://arxiv.org/html/2607.08780v1 、FRI-MxMoE https://aclanthology.org/2026.acl-long.982/ 。dispatch/combine分離・capacity管理はDeepSpeed-MoE／Tutel流、dropless不採用（CPU学習では固定capがマイクロカーネルタイル＋L2常駐192KB/expertを保つ）の判断は `analysis/gemm-design.md` §1–§2の通り。対照点としてLFM2（64 experts top-4）は粗い粒度であり我々の対照記録とする（`analysis/decoding-architecture.md` §2.1）。
- **未解決課題**：最終的なrouted数・top-k確定値は総量（§6）・active（§4）・NVMe 103MiB/token・起動≦50/tokenとの連立でP3b/cに申送る（本書は範囲固定であり単点固定はしない）。variance-aware capは条件未達のため未実施（steady-state>5%時の再評価条件として残す。`analysis/phase-g-summary.md` §6）。NEON実機・AVX-512移植は未実施（AVX2 Mr=12×Nr=4のみ確定）。
- **実装コスト（相対）**：中（Phase G Step 4 micro確立済みのためソート＋バッチングは中。top-k・容量・粒度の変更は設計改訂であり、マイクロカーネル再調整はsoak改訂扱い）。
- **依存**：`analysis/phase-g-summary.md`（G1–G4ゲート・Step 4新ベースライン・天井比58.4%・4.54x/5.74x）、`analysis/roofline.md` D6（expert固定・トークン走査、AI≈23の前提）、`analysis/training-speedup.md` P0-1/P0-2（ルーティング熱pin・I/O重ね合わせ・batch-union・PILOT相当）、`analysis/load-balancing-check.md`（μ効果確認済み・LB修正なし）。
- **ゲート条件**：perm/off/dropマスクのbit一致、drop率の§1.2トリガ評価（steady-state＝最後100 step平均。恒常的>5%で見直し検討、自動調整なし）、G2関数等価性（最終val差≦1%・勾配±10%）・G4増幅有界性（飽和<0.3%）の枠組みを維持すること。capacity・renormalize・μの変更は設計改訂とする。

## 4. アクティブ目標（A100M基準＋A300-500M選択肢の条件付き記述）

- **決定事項**：**A100Mを基準目標**とする（AGENTS.md §1・§8の「アクティブ100M」を維持）。内訳目安：100M INT4≒50MB＋縮小head INT8約12.7MB＋KV微小≒63〜100MB/token（`analysis/p3-design-inputs.md` §(a)A2・§(c)C1）。**A300-500Mは条件付き選択肢**とし、既定化しない。条件：(i)実効効率η≧0.3の中間点維持（§下記保守表）、(ii)KV微小維持（CSA2＋FP4＋top-1024・window128・heavy128:1）、(iii)未縮小headの除去済み（§5）、(iv)NVMe 103MiB/token内にpin-hit率＋Delta持続率で収めること（`analysis/p3-design-inputs.md` C1受入条件）。Ryzen保守上限約0.4B=400Mを500Mが超過するため、500M側はRyzen 30tok/s目標との両立に追加条件（W8A8 INT8化・バッチ和集合・PILOT相当の実測裏付け）を要する。
- **根拠（参照URL）**：帯域式 `tok/s≒eff_BW/bytes_per_token`（`analysis/p3-design-inputs.md` A1、`analysis/active-upper-bound.md` §1と同一）。較正はLFM2-24B-A2B（Q4_K_M約1.3GB/tokenで112tok/s→実効約145GB/s、η≒0.57。https://huggingface.co/LiquidAI/LFM2-24B-A2B 、https://www.liquid.ai/blog/lfm2-24b-a2b ）を存在証明に限定し、toks/s cross-vocab直接比較は行わない。JIMOTONO公称η=0.5は較正0.57より保守的。`analysis/roofline.md` D4（全カーネル帯域律速＝帯域式適用の正当化）。
- **保守値との整合（`analysis/active-upper-bound.md`準拠）**：公称上限（η=0.5・KV微小・オーバーヘッド無視の理論上限）N100約8B／Ryzen約2.6Bは設計目標に使わない。**上限目安には保守値（η=0.1＋KV/スケール/固定費で×0.8）：N100約1.3B・Ryzen約0.4Bを使う**。100M基準は保守値に対してもN100で約13×、Ryzenで約4×のマージンがあり、帯域は律速ではない（律速候補は未縮小head・起動固定費・NVMe IOPS/遅延・スレッドsync）。η=0.3中間点（N100約3.8B・Ryzen約1.3B）はA300-500M選択肢の可否判断の参照点とする。精度換算はINT4 0.5B/p（スケール込みQ4_K_M相当なら約0.56B/p）、INT8 1.0B/p、INT2 0.25B/p。混合精度の実効上限は加重平均で割り直すこと。
- **未解決課題**：η実測（STREAM未計測±30%）、単ch・小GEMV・競合時の下振れ、長文脈・大バッチ時のKV加算、O_DIRECT/SSD待ち混入、スレッドsync逓増（6T 6.2%→12T 11.3%。`analysis/f2-breakdown.md`）。いずれもP3実測で確定する（本書は式と方針のみ）。
- **実装コスト（相対）**：小（目標値自体のコストはない。コストは§5 head縮小・KV微小化・I/O重ね合わせ側に計上）。
- **依存**：`analysis/active-upper-bound.md` §2–§4、`analysis/p3-design-inputs.md` §(a)・§(c)C1、`analysis/vocab-budget.md`（§5）、`analysis/kv-cache-design.md`、`analysis/f2-breakdown.md`（sync・bwd効率）、AGENTS.md §8。
- **ゲート条件**：公称8B/2.6Bを受入条件・設計目標として引用しないこと。LFM2/Gemma値を同語彙速度比較に使わないこと。A300-500M選択肢の採用は上記(i)–(iv)＋η実測を満たした設計改訂でのみ行うこと。

## 5. 語彙 V=48,588＋factorized head（vocab-budget準拠・untie既定）

- **決定事項**：語彙はllm-jp-tokenizer v2.2の **V=48,588を維持**（変更なし）。d_model作業仮定=1024（proxy n=256と混同しない）。headは **factorized head（1024→256→48,588、k=256固定）を既定**とし、**emb full＋head factorizedのuntie**とする（「untieかつfull」でも「tiedのまま縮小」でもない）。精度既定INT8。head単体49.75M→12.70M（約3.92×減、−37.1MB/token。N100 5tok/sで−186MB/s、Ryzen 30tok/sで−1.1GB/s）。語彙全体既定はemb 49.75M＋head 12.70M＝約62.45M。emb交通費はd B/token（1024で1KB/token、無視級）。AGENTS.md §2.2の「tied embeddings」表記は本既定で上書きされる対象としてP3本体設計へ申送る（本wtでは表記変更しない）。
- **根拠（参照URL）**：語彙 https://github.com/llm-jp/llm-jp-tokenizer 、`analysis/tokenizer-gap.md`（V維持の根拠。C encoder現行3.648MB/s・100MB/s+目標は書換え級・優先度低・`.jtdp`再利用のため訓練クリティカルパス外）。算術は `analysis/p3-design-inputs.md` §(b)B1（受領値約50Mと一致）・`analysis/vocab-budget.md` §2–§3と同一。容量勘定（dense≦2GB／全体≦8GB）はemb INT8約50MB＋head INT8約12.7MBで予算内。
- **未解決課題**：対案factorized-tiedは不採用（共有のための追加射影が検証コスト増＋分離利点を消す。P3設計会議の記録事項）。head疎化（top-k logit等）の実装はP4（記録のみ）。untie化による+12.7M（tied比）の学習安定・量子化感度の実測はP3b以降。
- **実装コスト（相対）**：小〜中（2段GEMV化自体は小。untieに伴う学習・量子化の分離検証を含めると中）。
- **依存**：`analysis/vocab-budget.md` §1–§5、`analysis/p3-design-inputs.md` §(b)、`analysis/tokenizer-gap.md`、AGENTS.md §2.2・§8。
- **ゲート条件**：「約12.7Mはhead単体の値、語彙全体は約62.5M」と読むこと（混同禁止）。factorizedはheadのみ・k=256・emb full INT8常駐、既定untieを崩さないこと。head疎化の実装着手は禁止（P4記録のみ）。

## 6. 総10〜20B（固定）

- **決定事項**：モデル総量 **10〜20B**を維持する（AGENTS.md §1）。dense常駐（注意＋共有expert＋emb＋head）のINT4/INT8混成で2GB以内、全体8GB以内は§4・§5の縮小版としてdispatch可能（Colibri 17B→9.9GB実績の縮小版の位置づけ。https://github.com/JustVugg/colibri ）。混合精度はAGENTS.md §2.2を維持する（共有expert INT8・routed down INT4・routed gate/up INT2・注意射影INT8・emb/head INT8＋§5 untie既定）。FRI-MxMoE（https://aclanthology.org/2026.acl-long.982/ ）はエキスパート内サブレイヤーのロバスト性差の捉え方として参照する（プロファイリング高速化の輸入はP3b以降の判断）。
- **根拠（参照URL）**：混合精度の参照 https://huggingface.co/OsaurusAI/ZAYA1-8B-JANGTQ_K 、容量勘定 `analysis/p3-design-inputs.md` §(b)B2、帯域上限 `analysis/active-upper-bound.md`、W8A8 INT8本線（`analysis/training-speedup.md` §5。SGLang CPU https://docs.sglang.io/docs/hardware-platforms/cpu_server ）。T-MAC LUT https://ar5iv.labs.arxiv.org/html/2407.00088 はGEMM体制（prefill・学習）副線でありdecode GEMVではQLUT構築がamortize不可（`analysis/roofline.md` D4末尾・D6）。
- **未解決課題**：層数・expert数割付けの詳細はP3本体設計の入力とする（本書は上限と方針のみ）。10B超の常駐不能域での粒度ポリシー（安価recompute／重量offload＋fraction予約）は仕様化のみ（`analysis/training-speedup.md` P0-4）。INT2 gate/up・INT4 downの精度実測、HGQ-LUT学習→推論コンパイルの検証は未実施。
- **実装コスト（相対）**：大（総量スケールの実装はP3b/c本体であり、本書では上限と方針のみ固定する。P3aで実装着手しない）。
- **依存**：§1–§5、AGENTS.md §1・§2.2・§8、`analysis/training-speedup.md` P0-3/P0-4/P1-1/P1-2（W8A8本線・細粒度offload概念・MEFT疎更新・HyGIN式二経路fetch）、`analysis/f2-breakdown.md`（bwd 46%支配・optim 12%・sync逓増・ckpt/esmoe微小の基準値）、`analysis/roofline.md` D4–D7。
- **ゲート条件**：dense≦2GB／全体≦8GBの容量勘定を崩さないこと。量子化・融合・KV・decodeの変更をPhase G効果に上乗せ計上しないこと（非スコープ混入の `git status`／ビルドhash確認）。P3b/P3cへの着手は本書承認後に別タスクとすること。

## 参照URL一覧（本書の前提に限定）

- https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention
- https://arxiv.org/html/2605.22791
- https://github.com/NVlabs/GatedDeltaNet-2
- https://github.com/FareedKhan-dev/kimi-k3-in-c
- https://github.com/sehaxe/burn-kda
- https://aclanthology.org/2024.emnlp-main.916.pdf
- https://huggingface.co/FlameF0X/TinyMoE-100m-2x8
- https://deepseekv4.space/blog/deepseek-v4-technical-report
- https://arxiv.org/html/2607.08780v1
- https://aclanthology.org/2026.acl-long.982/
- https://github.com/llm-jp/llm-jp-tokenizer
- https://huggingface.co/OsaurusAI/ZAYA1-8B-JANGTQ_K
- https://ar5iv.labs.arxiv.org/html/2407.00088
- https://huggingface.co/LiquidAI/LFM2-24B-A2B
- https://www.liquid.ai/blog/lfm2-24b-a2b
- https://arxiv.org/html/2511.23404
- https://github.com/JustVugg/colibri
- https://docs.sglang.io/docs/hardware-platforms/cpu_server
- https://github.com/sgl-project/sglang/blob/4a50cd78/sgl-kernel/csrc/cpu/mamba/fla.cpp
