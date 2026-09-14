# P3学習スケール設計（P3c・学習のみ）

- branch: `docs/p3/training`（本wtのみ。コミットしない）
- 性質：文書のみ。実装禁止、重負荷実行なし（Web調査＋既存計測値の再利用のみ）。
- 前提：`analysis/p3-architecture.md`（P3a）が最優先であり、他すべての前提。本書はP3aと矛盾させない。矛盾発見時は本書を直さず設計会議へ申送る（Phase Gの再拡大は禁止）。
- 位置づけ：学習のみ。P3b（本体設計）・実装・Phase G再拡大には着手しない。4文書を混ぜない（推論・語彙・KV・デコードの再定義はしない。必要分は依存として輸入する）。
- コスト表記：相対コストは Phase G Step 4 micro（`jt_gemm_mat_f32`、Mr=12×Nr=4）を「中」とした相対（小＜中＜大）。絶対人月・絶対期間は記さない（P3a §冒頭と同一）。
- 数値の使い分け：帯域上限は公称値（N100 8B／Ryzen 2.6B）を設計目標に使わず、保守値（N100 約1.3B・Ryzen 約0.4B）を上限目安に使う。toks/sのcross-vocab直接比較は行わない（steps/s＋byte/stepに正規化）。
- 整合対象：`analysis/p3-architecture.md` §0–§6、`analysis/training-speedup.md` P0/P1、`analysis/p3-design-inputs.md` §(a)–§(c)、`analysis/f2-breakdown.md`、`analysis/roofline.md` D4–D7、`analysis/phase-g-summary.md`、`analysis/active-upper-bound.md`、`analysis/vocab-budget.md`、`analysis/kv-cache-design.md`、`analysis/tokenizer-gap.md`、AGENTS.md §4・§5・§8。

## 1. W8A8＋Megatron細粒度offload（training-speedup.md P0準拠）

- **決定事項**：
  - W8A8 INT8 MoE経路を学習・prefillのGEMM本線とする（P0-3）。ブロック量子化＋AMX／AVX2-VNNI／NEON-I8MM分岐。帯域半減をP3見積りの既定係数にする。AGENTS §2.2混合精度表（共有expert INT8・注意射影INT8・routed down INT4・routed gate/up INT2・emb/head INT8＋P3a §5 untie既定）と整合し、INT8本線が低ビット表を置き換えない。
  - T-MAC LUT（INT2/INT4 routed）は副線に限定する。GEMM体制（Mが大きいprefill・学習）でのみ有効とし、decode GEMVには混入させない（`analysis/roofline.md` D4末尾・D6と同一）。
  - Megatron細粒度offloadは粒度ポリシー（概念）のみ輸入する（P0-4）。対象サブモジュール表をCPU学習に写像する：`attn_norm`・`qkv_linear`・`core_attn`・`attn_proj`・`mlp_norm`・`expert_fc1`・`moe_act`・`fused_group_mlp`（GDN-2層ではチャンクWY学習側を`core_attn`相当に割付ける。デコード逐次再帰側には混入させない）。
  - 使い分け則（公式ポリシー準拠）：**安価モジュール（layernorm系・`moe_act`・RMSNorm/RoPE融合部・gate logits→topk→softmax小物）はrecompute、重量モジュール（`core_attn`・`expert_fc1`・`fused_group_mlp`・GDN-2チャンクWY）はoffload（退避先RAM↔SSD）**。`offload-fraction`相当ノブを予約する（既定値固定なし）。
  - 28M常駐域では発動しない。`analysis/f2-breakdown.md`実測（ckpt 5.7ms/step・esmoe_io 0.7–0.9ms/step、stepの0.3%以下で律速でない）より、本手法は「総10〜20Bで常駐不能になったときの粒度ポリシー」として仕様化のみ行う。
- **根拠（参照URL）**：SGLang CPUサーバ文書 https://docs.sglang.io/docs/hardware-platforms/cpu_server 、Megatron細粒度offload https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/fine_grained_activation_offloading.md 、解説 https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/fine_grained_activation_offloading.html 、T-MAC https://ar5iv.labs.arxiv.org/html/2407.00088 、混合精度参照 https://huggingface.co/OsaurusAI/ZAYA1-8B-JANGTQ_K 、FRI-MxMoE https://aclanthology.org/2026.acl-long.982/ （サブレイヤー別ロバスト性の捉え方のみ）。
- **未解決課題**：GDN-2学習カーネル（チャンクWY非対称erase因子＋gate-aware backward）のC学習エンジン側コスト未確定（P3a §1と同一）。FP4/INT8中間活性の精度未検証（層単位FP8フォールバック要否はP3b以降）。CUDAグラフ内にoffload対象を含められない暫定制限のCPU版（io_uring＋compute重ね合わせ時のordering）未設計。offload-fraction既定値・PP各種との対応は10B割付確定後に決める。
- **実装コスト（相対）**：中（W8A8 INT8経路の分岐整備で中。粒度表の仕様化のみは小。合計して中とし、大にはしない）。
- **依存**：`analysis/p3-architecture.md` §1・§6、`analysis/training-speedup.md` P0-3/P0-4/P1-2、`analysis/roofline.md` D4（全カーネル帯域律速）・D6（GEMM化AI≈23）・D7（F1–F3は1%級、G1はdecode側）、`analysis/f2-breakdown.md` §1–§2（ckpt/esmoe微小の基準値）、`analysis/phase-g-summary.md`（G効果の上乗せ計上禁止）。
- **ゲート条件**：28M域でのoffload発動禁止（実測で律速確認後のみ発動）。recompute/offload境界・fraction既定値の変更は設計改訂とする。T-MAC LUTをdecode GEMV体制に混入させないこと。W8A8帯域半減をP3見積りの既定係数として使うこと。

## 2. 熱記録＋ホットpin（.coli_usage相当の機構設計）

- **決定事項**：
  - Colibri学習キャッシュをP3学習エンジンに直輸入する（P0-1）。ルーティング選択実績をモデル脇sidecar（`.coli_usage`相当）に毎step追記し、起動時にホットexpertを自動pinする。記録単位は`(layer, expert_id, token_pos, slot)`辞書式順序（P3a §3のperm順序と同一規約）とし、独自バイナリ形式（AGENTS §1設計思想）の形式仕様に含める（配置プランナ`plan/doctor/tune`相当と同時設計）。
  - pin運用はColibri則を踏襲する：`PIN=auto`相当の自動pin＋`PIN_GB`相当の予算上限、適応LRUと予算共有するが**自動pinはLRU容量を侵さない上限付き**、`--repin N`相当の安全なstep境界での入替（減衰ヒートマップ、25%ヒステリシス、4スワップ上限でスラッシング防止）。
  - 更新系はprimary側に集約する（SmartUpdateのSSD上optimizer直接更新と同型）。読込専用複製（Dual-SSD相当）には書かない。読込失敗はprimaryへ縮退する。`O_DIRECT`は実測採用（ドライブ依存のため一律ONにしない。Colibri公式方針と同一）。
  - I/O重ね合わせ（P0-2）と併用する：expert 3行列の隣接配置＋1 `pread`読込、async pool（PIPE相当、matmulとpread重ね合わせ）、Linux `io_uring`（`PIPE=1`含意）、batch-union（同一expertは全positionで1回だけ読む）、PILOT相当lookahead（次層ルーティング予測。71.6%値は前提とせずP3で実測再取得）。サイズ別二経路（HyGIN式：小→同期pread／大→io_uringバッチ）はP1-2としてfetch経路に織り込む（論文値の転載はしない）。
  - StickyMoE（soft-hard variant）のルーティング熱をES-MoE pinに直結する。MEFT疎更新（P1-1：発火expertのみSSD上optimizer step）は本機構の拡張として位置づけ、実装は別phaseとする。
  - Dual-SSD・NUMA・MTPはP2に据え置く（速度オプション。MTPはint8 head必須の制約付きのため学習pin設計には混ぜない）。
- **根拠（参照URL）**：Colibri正準 https://github.com/JustVugg/colibri 、tuning https://github.com/JustVugg/colibri/blob/main/docs/tuning.md 、benchmarks https://github.com/JustVugg/colibri/blob/main/docs/benchmarks.md 、配布例 https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp 、StickyMoE https://arxiv.org/html/2607.08780v1 、ES-MoE https://ina.kaist.ac.kr/projects/esmoe/ 、HyGIN https://www.computer.org/csdl/journal/ca/2026/02/11601081/2hZ80QAVT8Y 、MEFT https://aclanthology.org/2024.acl-long.129 ・ https://arxiv.org/abs/2406.04984 ・ https://github.com/CURRENTF/MEFT 。
- **未解決課題**：sidecar記録フォーマット詳細（64Bキャッシュライン整列・ヘッダ検証・減衰係数）未仕様化。ヒステリシス25%・4スワップ上限の自前妥当性未検証。`O_DIRECT`のドライブ別実測未実施。PILOT 71.6%の自前再取得未実施。`.coli_kv`相当（会話またぎKV永続）の学習時要否はP3b以降（KV設計は`analysis/kv-cache-design.md`に委譲し本書で再定義しない）。
- **実装コスト（相対）**：小〜中（記録＋pin本体は小。repin安全境界＋非同期重ね合わせ＋batch-unionを含めると中）。
- **依存**：`analysis/training-speedup.md` P0-1/P0-2/P1-1/P1-2、`analysis/p3-architecture.md` §3（細粒度・top-k・cap=1.5・renormalize・μ=0.01・counting sort決定論性）、`analysis/load-balancing-check.md`（μ効果確認済み）、AGENTS §4.1（ckpt/esmoe/SmartUpdate）・§4.2（SwiGLU融合・O_DIRECT＋io_uring・S∪L∪M和集合）。
- **ゲート条件**：pin-hit率を自前実測で示すこと（Colibri配布例6,592,176 selectionで56.9%→72.8%約16pt改善を引用ではなく自前計測で再取得）。自動pinがLRU容量を侵さないこと。スラッシング防止上限（ヒステリシス＋スワップ上限）を守ること。toks/sのcross-vocab比較に使わないこと（GB/sとbyte/tokenに正規化）。

## 3. スケール計画：1B → 10B → LLM-jp Corpus v4

- **決定事項**：3段階に固定する（P3aの総10〜20B／A100M基準・dense≦2GB／全体≦8GB・起動≦50/token・NVMe≦103MiB/token・N100 5tok/s・Ryzen 30tok/sを変更しない）。
  - **S1＝1B確立**：`results/bench/20260914-1b-scale.md`構成を引き継ぐ（L=24・d=1024・E=64/layer・expert_hidden=128・shared=2・top-k=4・GDN 5:1、total 0.712B［routing 604M／shared 18.9M／attn 37.8M／emb 49.8M／router 1.6M］、active 96.018M＜100M、dense INT4 69.6MiB＜2GB、SSD experts 264MiB）。語彙V=48,588維持＋factorized head（1024→256→48,588、k=256固定）・emb full＋head factorizedのuntie既定・INT8（P3a §5）。
  - **S2＝10B拡大**：総量下限10B（P3a §6の10〜20B内）。1エキスパート2〜4M・routed数千・top-8〜16・共有2（P3a §3の範囲固定。単点固定はP3b/cに申送り）。1Bの約14倍（0.712B→10B）をoffload＋ckpt＋8bit optimizer＋MEFT疎更新で吸収する。
  - **S3＝LLM-jp Corpus v4本格**：日本語6,880億＋コード218B（AGENTS §5）を段階投入する。フル688Bの一括学習は単機不可のため、ja_wiki 2.2G級サブセット→SwallowCode-v2→Japanese Function Calling Datasetの順で拡大する（合成タスクのval 27%頭打ち脱却が目的。`results/bench/20260914-phased-val.md` held-out所見と同一）。
  - **16GB／N100可否**（既測値のみで判定。楽観外挿をしない）：
    - S1学習peak（B=8/S=2048/fp32・ckpt 5区間＋optim8＋offload）1.937GiB→16GBに収まる。N100（11GB）判定：可（offload推奨。なしでも7.54GiBだがマージン薄）。
    - S1推論fetch約16MiB/token（24層×top4×172KiB）→103MiB/token基準の内側に入り得る水準（未計測要素多し）。
    - S2学習の単機（16GB）成立は§1粒度ポリシー＋8bit optimizer（状態1/3.8）＋発火分のみstepの併用が条件。N100でのS2学習実実行は不可（学習step時間未実測のため予測不可と明記）。
    - S2推論のN100/Ryzen可否はP3a §4の条件付き選択肢則に従う（η≧0.3中間点・KV微小・未縮小head除去済み・NVMe 103MiB内にpin-hit＋Delta持続で収めること）。
    - S3データ取得は100MB超DL未実行の現状を維持する（実DLはP3b以降の計画実行時に行う。本書では取得しない）。
- **根拠（参照URL）**：LLM-jp Corpus v4 https://github.com/llm-jp/llm-jp-corpus-v4 、語彙 https://github.com/llm-jp/llm-jp-tokenizer 、SwallowCode-v2 https://huggingface.co/datasets/tokyotech-llm/swallowcode-v2 、Japanese Function Calling Dataset https://huggingface.co/datasets/Tonari-no-usagi/Japanese_Function_Calling_Dataset 、Colibri https://github.com/JustVugg/colibri （縮小版dispatchの位置づけ）、共有expert・CSA https://deepseekv4.space/blog/deepseek-v4-technical-report 、細粒度 https://huggingface.co/FlameF0X/TinyMoE-100m-2x8 。
- **未解決課題**：10Bの層数・expert数割付け詳細はP3本体設計の入力（本書は上限と方針のみ）。10B超でのoffload発動点・fraction既定値未定。N100学習時間の実測なし（microbench regime差のため予測不可）。Corpus取得・ライセンス・前処理（`.jtdp`再利用。C encoder現行3.648MB/s・100MB/s+目標は書換え級のため訓練クリティカルパス外。`analysis/tokenizer-gap.md`と同一）未着手。lrスケジュール未実装（d=1024級で1e-3発散・2e-4完走のみ確認）。
- **実装コスト（相対）**：小（本書は段階・ゲート定義のみ。実装本体はP3b/cで大。P3a §6「総量スケール実装はP3b/c本体」と同一の分離）。
- **依存**：`analysis/p3-architecture.md` §0・§4・§5・§6、`analysis/p3-design-inputs.md` §(b)B2・§(c)C1、`analysis/active-upper-bound.md` §2–§4、`analysis/vocab-budget.md`、`analysis/kv-cache-design.md`（フル層のみCSA2＋FP4）、`analysis/tokenizer-gap.md`、`analysis/phase-g-summary.md`（効果の上乗せ計上禁止）、AGENTS §5・§8。
- **ゲート条件**（段階改訂は設計改訂扱い。場当たり調整禁止）：

| 段階 | entry（入場条件） | exit（次段への合格条件） |
|---|---|---|
| S1 1B | model1b会計（total/active/dense/SSD）・gate勾配finite-diff・N150 15/15 pass・warnings 0を維持すること | 学習peak≦2GB実測・fetch≦103MiB実測・drop steady-state（最後100 step平均）≦5%・Step 4 micro基準回帰なし（最終val差≦1%・飽和＜0.3%）・η実測の記録 |
| S2 10B | S1 exit通過＋§1粒度ポリシー＋§2 pin機構の仕様凍結＋offload-fractionノブ予約 | dense≦2GB／全体≦8GB・A100M基準維持（A300-500MはP3a §4の(i)–(iv)＋η実測を満たす設計改訂でのみ）・NVMe≦103MiB・起動≦50/token・容量勘定の維持 |
| S3 Corpus | S2 exit通過＋実データ前処理（ja_wiki subset→SwallowCode-v2→JFCD）の順序凍結 | 合成val頭打ち（比0.73・27%改善）からの有意改善・日本語MT-Bench／SWE-bench Pro／ツールコール成功率（AGENTS §8）はP3本体評価で判定（本書で合否を付けない） |

## 4. RL：ORPO／SPO＋GRPO／RLVR（AGENTS §4.3準拠）

- **決定事項**：順序と前提条件を以下に固定する（AGENTS §4.3の表を変更しない）。
  - 順序：**(1) SFT → (2) ORPO（SFTと同時。参照モデル不要。順伝播1回） → (3) SPO（過最適化抑制。順伝播1回） → (4) GRPO（KL削除。生成N回＋更新1回）＋RLVR（コード実行・ツールコール成否のルールベース報酬）**。SimPO（選好学習。順伝播1回）はORPO/SPOの代替肢とし、三重同時適用はしない（過最適化抑制と選好学習の重複適用は設計改訂扱い）。
  - 前提条件：ORPO/SPOはSFTデータ＋選好ペア（Japanese Function Calling Dataset等のツールコール選好を含む）が前提。GRPO/RLVRは( i )コード実行サンドボックス＋function-call成否判定器、( ii )生成N回分の計算・メモリ・KV予算（CSA2＋FP4＋top-1024・window128・heavy128:1でKV微小を維持）、( iii )W8A8推論本線の確定が前提。GRPOの先行開始は禁止する（SFT＋ORPO/SPOの通過がentry条件）。
  - RLVR報酬はルールベースのみ（コード実行可否・ツールコール成否）。学習済み報酬モデルの導入は設計改訂とする。
- **根拠（参照URL）**：AGENTS §4.3表（GRPO不要KL削除・生成N回＋更新1回／ORPO不要・1回／SimPO不要・1回／SPO不要・1回）、nanoRL https://github.com/alex000kim/nanoRL 、データセット https://github.com/llm-jp/llm-jp-corpus-v4 ・ https://huggingface.co/datasets/tokyotech-llm/swallowcode-v2 ・ https://huggingface.co/datasets/Tonari-no-usagi/Japanese_Function_Calling_Dataset 。
- **未解決課題**：日本語選好ペアの調達量未確定。報酬ハック対策未設計。KL削除時の発散抑止におけるSPO先行の効果未検証。生成N回時のoffload・pin競合（学習fetchと生成fetchの干渉）未評価。RL開始段階（S1後半〜S2の提案）は未確定でありP3本体設計の入力とする。
- **実装コスト（相対）**：小〜中（ORPO/SPOの1-pass適用は小。GRPOのN生成＋RLVR実行環境を含めると中。大にはしない）。
- **依存**：AGENTS §4.3・§5、`analysis/p3-architecture.md` §4（A100M基準）・§6（混合精度）・§2（比率固定）、`analysis/training-speedup.md` P0-3（W8A8本線）・P0-1/P0-2（fetch重ね合わせ）、`analysis/kv-cache-design.md`（生成時KV微小の前提）。
- **ゲート条件**：順序逆転禁止（GRPO/RLVRの先行開始禁止）。参照モデル不要性の維持（参照モデル導入は設計改訂）。順伝播回数（ORPO/SPO/SimPO＝1回、GRPO＝生成N回＋更新1回）の遵守。RLVR報酬はルールベースのみ。成功基準（日本語MT-Bench・SWE-bench Pro 50・ツールコール成功率90%。AGENTS §8）の合否判定は本書で行わない（P3本体評価に委譲）。

## 5. 学習スループット目標（現状実測を起点に各段階の目標を設定）

- **決定事項**：起点実測を以下に固定し（いずれも既存記録の再利用。新規測定なし）、段階目標は相対表現＋割付けで設定する。絶対toks/s目標の固定はしない（STREAM未計測±30%のため絶対外挿禁止）。
  - 起点A＝5M-proxy longrun：0.363 steps/s・186 toks/s・7000s flat（layers=6・d=256・E=16・h=64・top-2・shared=1・seq=128・batch=4・fp32・5,038,337 params・約323MiB resident・12 threads。`results/bench/20260914-longrun.md`）。実学習ループ（moe fwd/bwd＋gate勾配＋optim8＋ckpt＋offload）の2時間安定動作の基準。
  - 起点B＝5M-proxy AVX2 peak：1270 toks/s（2T peak。scalar12T 451→AVX2 12T 474で帯域飽和、1T 435→852で1.96x。baseline比2.8x。`results/bench/20260914-phased-val.md`）。1B外挿1270÷140≈9 toks/sは楽観側・帯域無視の参考値に限定する。
  - 起点C＝100M-proxy：33.4 toks/s（0.521 steps/s・64 toks/step。L=4/d=1024/E=16/h=128/K=4/S=2/seq=32/batch=2・28.4M params・1.9GB resident。`results/bench/20260914-phasef.md`）。内訳は`analysis/f2-breakdown.md` 6T値：bwd 872ms 46%・fwd 113ms 6%・optim 222ms 12%・sync 117ms 6%（12Tで231ms 11%へ逓増）・ckpt 5.7ms 0.3%・esmoe_io 0.8ms 0.04%・val償却539ms 29%（1 event約2.15s。測定artifactであり定常除外。除外時の真の学習は1347ms/step＝47.5 toks/s@6T）。
  - 起点D＝TinyStories micro（Step 4 micro新ベースライン。`analysis/phase-g-summary.md` §3）：1000 steps micro 30.58 steps/s（32.7s。T=512→15640 toks/s）／single 6.68 steps/s（149.8s。比4.58xは参考値）。500 steps micro 30.49 vs single 6.71＝4.54x、longrun micro 4.748 vs single 0.827＝5.74x、天井比58.4%（＞50% pass）。dims d=64・layers=2・E=8・K=2・V=258のため他起点とのtoks/s比較は禁止し、steps/s倍率のみ輸入する。
  - 段階目標（`analysis/p3-design-inputs.md` §(c)C3の割付けをP3c目標として確定する）：
    - S1：bwd 46%→半減（融合SwiGLU gate/up/down 1カーネル＋RMSNorm/RoPE融合＋W8A8本線）、optim 12%→6%以下（8bit optimizer状態1/3.8＋MEFT疎更新＝発火expertのみstep）、sync逓増の抑制（スレッド数適正化＋計算／通信／optimizer重ね合わせで6T比横ばい）、val固定費の除去（頻度・checked-API見直しで償却431ms/step→50ms/step以下相当。学習カーネルではないが現行第2律速のため必須）、ckpt/esmoeは現0.3%以下の維持（S1では発動させない）。
    - S2：offload発動域でS1効率の50%以上維持（計算／通信／optimizer重ね合わせ＋offload-fraction予約範囲内）。F1–F3融合（全体1%級）は重み削減後にのみ評価する（`analysis/roofline.md` D7と同一）。
    - S3：toks/s絶対値ではなくval改善率＋単位GB/sあたり学習効率で判定する（合成27%頭打ちの脱却）。
    - 全段階でPhase G効果の上乗せ計上禁止（非スコープ混入の`git status`／ビルドhash確認を維持）。
- **根拠（参照URL）**：起点A `results/bench/20260914-longrun.md`、起点B `results/bench/20260914-phased-val.md`、起点C `results/bench/20260914-phasef.md`＋`analysis/f2-breakdown.md`、起点D `analysis/phase-g-summary.md` §3–§4、割付け `analysis/p3-design-inputs.md` §(c)C3、帯域律速 `analysis/roofline.md` D4–D7、W8A8 https://docs.sglang.io/docs/hardware-platforms/cpu_server 。
- **未解決課題**：STREAM未計測（eff_BW=29.4GB/sは推定±30%。絶対値外挿に±30%を見込む）。N150実値（L3 6MB）と前提差異。bwd効率がfwdの約1/5（実効約4.6GB/s対21GB/s、2.75 GFLOP/s対10.6 GFLOP/s）の原因未特定（H=128小行列のベクトル化・依存連鎖・スパース加算パターンの疑い）。1B学習1ステップ時間未実測（microbench regime差のため予測不可）。lrスケジュール未実装。
- **実装コスト（相対）**：小（目標値自体のコストはない。コストは§1融合・W8A8・I/O重ね合わせ側に計上。`analysis/p3-design-inputs.md` C3と同一）。
- **依存**：`analysis/f2-breakdown.md` §4–§6、`analysis/roofline.md` D4–D7、`analysis/phase-g-summary.md` §3（Step 4 micro新ベースライン・追跡基準値）・§5（cap=1.5・renormalize・μ=0.01）・§6、`analysis/p3-design-inputs.md` §(c)C3、`analysis/training-speedup.md` P0/P1、`analysis/p3-architecture.md` §4・§6。
- **ゲート条件**：S1 exit＝bwd wall半減実測＋val固定費除去実測＋optim半減実測＋sync横ばい実測＋drop steady-state≦5%＋Step 4 micro基準回帰なし（最終val差≦1%・飽和＜0.3%・perm/off/drop bit一致）。公称8B/2.6Bを受入条件・設計目標として引用しないこと。LFM2/Gemma値を同語彙速度比較に使わないこと。絶対toks/s目標の固定はSTREAM実測後に設計改訂で行うこと。

## 追補A. RL順序の意図（ORPO＋SPO両用の理由・実験計画とパイプラインの分離・P3d確認記録）

- **ORPO＋SPO両用の理由（誤記ではない）**：ORPO＝選好学習（odds-ratio、SFTと同時・順伝播1回）、SPO＝過最適化抑制（順伝播1回）であり役割が異なるため、順次適用 (1)SFT → (2)ORPO → (3)SPO → (4)GRPO＋RLVR を既定とする。SimPO（選好学習・1回）はORPOの代替肢であり、ORPO／SimPO／SPOの三重同時適用は重複のため禁止（設計改訂扱い）。ROADMAP Phase 4の「ORPOまたはSPO」表記は計画段階の選択肢表現であり、本書の順次既定と矛盾するためROADMAP改訂で「ORPO→SPO順次・SimPO代替」に統一する（P3d申送り）。
- **パイプライン設計（コード経路）**：SFT損失／ORPO 1-pass／SPO正則化／GRPO N生成＋更新1回＋RLVRサンドボックス（コード実行・ツールコール成否のルールベース報酬のみ、報酬モデル導入は設計改訂）。GRPO先行開始禁止（SFT＋ORPO/SPO通過がentry条件）、参照モデル不要性の維持。
- **実験計画（評価順序・本体設計への入力）**：S1後半〜S2でORPO/SPOの効果（選好改善＋過最適化抑制）を検証し、GRPO/RLVRは (i)サンドボックス＋判定器、(ii)生成N回分の計算・メモリ・KV予算（CSA2＋FP4微小維持）、(iii)W8A8本線確定を満たした後に開始する。成功基準（MT-Bench・SWE-bench Pro 50・ツールコール90%）の合否は本書で付けずP3本体評価に委譲する。

## 参照URL一覧（本書の前提に限定）

- https://github.com/JustVugg/colibri
- https://github.com/JustVugg/colibri/blob/main/docs/tuning.md
- https://github.com/JustVugg/colibri/blob/main/docs/benchmarks.md
- https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp
- https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/fine_grained_activation_offloading.md
- https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/fine_grained_activation_offloading.html
- https://docs.sglang.io/docs/hardware-platforms/cpu_server
- https://aclanthology.org/2024.acl-long.129
- https://arxiv.org/abs/2406.04984
- https://github.com/CURRENTF/MEFT
- https://www.computer.org/csdl/journal/ca/2026/02/11601081/2hZ80QAVT8Y
- https://ar5iv.labs.arxiv.org/html/2407.00088
- https://huggingface.co/OsaurusAI/ZAYA1-8B-JANGTQ_K
- https://aclanthology.org/2026.acl-long.982/
- https://arxiv.org/html/2607.08780v1
- https://ina.kaist.ac.kr/projects/esmoe/
- https://github.com/alex000kim/nanoRL
- https://github.com/llm-jp/llm-jp-corpus-v4
- https://github.com/llm-jp/llm-jp-tokenizer
- https://huggingface.co/datasets/tokyotech-llm/swallowcode-v2
- https://huggingface.co/datasets/Tonari-no-usagi/Japanese_Function_Calling_Dataset
- https://huggingface.co/FlameF0X/TinyMoE-100m-2x8
- https://deepseekv4.space/blog/deepseek-v4-technical-report

## 禁止事項の遵守記録

- P3b・実装・Phase G再拡大：なし（本書は学習スケールの段階・ゲート・割付け定義のみ）。
- 4文書混交：なし（推論・語彙・KV・デコードの再定義をせず依存輸入に留めた。語彙スパース化の実装着手なし）。
- toks/s cross-vocab比較：なし（帯域・byte/token・steps/s正規化のみ）。
- Phase G効果の上乗せ計上：なし（Step 4 microを追跡基準として固定し回帰判定にのみ使う）。
- 重負荷実行：なし（既存計測値の再利用のみ。新規bench・DL・学習実行なし）。
- コミット：しない（作業ツリー内文書のみ）。
