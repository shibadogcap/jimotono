# デコードアーキテクチャ — KDA / LFM2-24B-A2B / SGLang CPU 調査

- branch: `docs/p3/survey-kv-decode`（本wtのみ。コミットしない）
- 位置づけ: P3設計入力。実装着手は禁止（調査＋文書のみ）。
- 前提: 線形注意はGated DeltaNet-2採用済み（AGENTS.md §2.1、ARCHITECTURE.md §1）。
- 禁止事項遵守: P3実装なし、語彙スパース化の実装なし、異語彙間のtoks/s比較なし、「CPU非現実」での打切りなし。

## 結論（先に）

1. **P3の線形層はGated DeltaNet-2を維持。KDAは「縮退版フォールバック＋デコード実装の参照」として位置づける。** GDN-2はKDAの厳密な一般化であり（erase/write両ゲートを同一スカラに束縛するとKDAに還元）、KDA採用はGDN-2採用の部分集合になる。後退の選択肢はない。
2. **kimi-k3-in-c（C99）のデコード実装は我々のC移植の直接の手本になる。** `k3_kda_step`は素朴な逐次再帰（チャンク並列なし）、固定サイズ状態の持越し＋KVキャッシュの増分デコード、3段階ゲート検証。学習用チャンクワイズ（UT変換逆行列等）は意図的に未実装で文書側に隔離——この分離判断自体がP3の手本。
3. **LFM2-24B-A2Bの教訓は「比率の裏付け」と「MoE粒度の対照」であり、速度値の横並び比較はしない**（語彙・量子化・測定条件が異なるため）。30 conv : 10 attn（1:3）のハイブリッドで実用速度が出る事実は、我々の4:1〜5:1（線形寄り）が現実的範囲内であることの傍証。
4. **SGLang CPUバックエンドはアルゴリズム構造の参照先。** 直接リンクはAGENTS.md §7.2で禁止のため、`fla.cpp`の融合順序・チャンク/再帰の切替・conv1d更新パターンのみをC再実装の設計入力とする。

## 1. KDA（kimi-k3-in-c、C99実装の精読）

対象: https://github.com/FareedKhan-dev/kimi-k3-in-c
（2.78TパラメータKimi K3を単一CPU・8.24GB RSSで推論。C99、BLAS/ framework/GPUなし）

### 1.1 モデル構成（config読解より）

- 93層（layer 0はdense FF、残り92層がMoE）。**69層KDA＋24層Gated MLA**（4層毎: 4,8,...,92,93。最終層は必ず全体注意）。
- KDA: 96 heads × head_dim 128（d_k == d_v）、conv_k=4（depthwise causal、SiLU融合）、gate_lb=-5.0。
- MoE: 896 routed（top-16）＋2 shared、latent 3584、vocab 163840。
- routed expertsはMXFP4（0.5B/weight＋共有E8M0スケールで0.53125B）。1.447TB/1.56TBがrouted experts。

### 1.2 固定ステート更新（デコード）

- `k3_kda_step`は**素朴な逐次再帰を1位置ずつ実行**。チャンク並列形（UT変換の逆行列 `(I+Akk)^-1`、`Aqk`の対角保持等）は「実装していない主張」としてコードから除去し、`docs/ARCHITECTURE.md`側に隔離。チャンクパス復活時はそれを縛るfixtureごと復活させる方針が明記されている。
- **KDA層は記憶が増大しない固定サイズ再帰状態**（"attention with a memory that never grows"）。増分デコードは「KVキャッシュ＋持越しKDA状態」で等価性検証。
- 検証の梯子（weightless tests）: GATE1 teacher forcing 32/32、GATE2 greedy decode 20/20、GATE3 incremental 20/20。「ENGINE MATCHES THE REFERENCE EXACTLY」。
- 数値契約: `-ffp-contract=off`でスカラ/OpenMP/AVX2パスの**ビット同一**を保証（性能変更が精度変更に化けないための措置）。
- 不変条件3件がP3にも有用: (1)`A_log`はhead毎index（後半はpadding）、(2)MLAはNoPEだが64 rope次元スロットは保持・cacheする（回転のみ省略）、(3)MoE routing biasは選択のみに使い結合重みはunbiased sigmoidから。

### 1.3 KDA更新式（GDN-2論文・burn-kdaより裏取り）

- コア再帰: `S_t = (I − β_t k_t k_t^T) Diag(α_t) S_{t−1} + β_t k_t v_t^T`、`o_t = S_t^T q_t`
  https://github.com/NVlabs/GatedDeltaNet-2
  https://github.com/sehaxe/burn-kda
- チャネル単位忘却（GDNのスカラ減衰に対する改良）＋能動編集は単一スカラβに束縛——この束縛こそGDN-2が解いた点（§3）。
- 工学的詳細（P3への借用候補）: q/k = L2Norm(Swish(ShortConv))、v = Swish(ShortConv)、head毎データ依存β=sigmoid、sigmoid有界減衰（g_min=-5、α∈(e^-5,1)）、full-rank出力ゲート、学習はチャンクWY形（GDN-2と同一代数、対応 b=β・g=log α・w=β）、f32でchunk 16が安定。

### 1.4 チャンクワイズ有無

- **デコード用Cエンジンにチャンクワイズなし**（逐次のみ）。学習用チャンクWY形はPyTorch参照・Triton融合カーネル側の話。我々のC推論エンジンもデコードは逐次再帰で足り、チャンク形は学習エンジン側の責務と切り分ける判断が成立する。

## 2. LFM2-24B-A2B

- https://huggingface.co/LiquidAI/LFM2-24B-A2B
- https://www.liquid.ai/blog/lfm2-24b-a2b
- https://arxiv.org/html/2511.23404 （LFM2技術報告）

### 2.1 比率・MoE粒度

| 項目 | LFM2-24B-A2B |
|---|---|
| 総/活性 | 24B / 2.3B |
| 層 | 40層（gated short conv 30＋GQA 10、**1:3**） |
| d_model / conv kernel | 2048 / k=3 |
| MoE | 64 experts、top-4、per-expert FF 1536（8B版は32 experts/top-4/1792）。先頭2層dense |
| コンテキスト/語彙 | 32,768 / 65,536 |
| 学習 | 17Tトークン、BF16/FP8混合 |

- 要点: hardware-in-the-loop探索の結果、**SSM/線形注意なし**（gated short conv多数＋少数GQA）で十分と判断した設計。conv層は固定小状態でprefill高速・省メモリ。
- MoE粒度は我々（2〜4M×数千、top-8〜16）より粗い（64 experts）。対照点として記録。

### 2.2 112 tok/sの内訳（横並び比較はしない）

- 公称: AMD CPUでdecode **112 tok/s**、H100で293 tok/s。条件はllama.cpp・Q4_K_M・Ryzen AI Max+ 395（モデルカード記載）。
- 第三者実測（同CPU、Vulkan backend、文書要約7,624トークン）: 生成109 tok/s、prefill 1,190 tok/s。
  https://fromthematrix.dev/posts/local-llm-speed-benchmark-strix-halo/
- vLLM・H100・1024並列: 合計約26.8K tok/s（競合MoEを上回る）。
- **扱い:** 語彙（65,536 vs 我々48,588）・量子化・測定条件が異なるため、我々の目標（N100で5 tok/s等）との優劣比較は行わない。本調査での利用価値は「conv:attn=3:1のハイブリッドでCPU実用速度が出る」という**比率の現実性傍証**に限定する。

## 3. SGLang CPUバックエンド（GDN/KDA/Mamba2）

- 文書: https://docs.sglang.io/docs/advanced_features/attention_backend
  （`--linear-attn-backend`、phase別override可。Triton(CPU)はdecode✅/prefill✅/spec❌）
- CPUカーネル: https://github.com/sgl-project/sglang/blob/4a50cd78/sgl-kernel/csrc/cpu/mamba/fla.cpp
  （`chunk_gated_delta_rule_kernel_impl`、`fused_sigmoid_gating_delta_rule_update`、ATen Vectorized、L2-norm融合、並列ループ削減）
- Python binding: `sgl-kernel/python/sgl_kernel/mamba.py`（`causal_conv1d_fn_cpu`/`update_cpu`、`chunk_gated_delta_rule_cpu`）
- backend: `python/sglang/srt/layers/attention/linear/gdn_backend.py`
  （`GDNKernelDispatcher`/`KDAKernelDispatcher`が`LinearAttnKernelBase`をdispatch。ハイブリッドのfull層は通常attention backendが別途必要）
- P3への借用は**構造のみ**: チャンク（prefill/extend）⇄融合再帰更新（decode）の切替、packed decode、conv状態更新の順序。コード・ライブラリの直接利用は禁止（AGENTS.md §7.2）。

## 4. KDA vs GDN-2 比較表＋P3採用決定

GDN-2根拠: https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention
（arXiv全文: https://arxiv.org/html/2605.22791 、実装: https://github.com/NVlabs/GatedDeltaNet-2）

| 軸 | KDA | GDN-2 | 備考 |
|---|---|---|---|
| 更新式 | 減衰channel-wise、編集スカラβ束縛 | **erase gate b_t（key側）＋write gate w_t（value側）をchannel-wise分離**、減衰channel-wise | b=w=β·1に束縛でKDAに還元、減衰も束縛でGDNに還元（厳密な一般化） |
| デコード速度 | O(1)/token固定状態 | 同一オーダー（ゲート投影2本増の定数増のみ） | 論文: small constant throughput overhead。支配項は同一 |
| 学習コスト | チャンクWY＋通常backward | チャンクWY（非対称erase因子吸収）＋gate-aware backward | GDN-2が若干重いが同体制。C推論側は無関係（逐次のみ） |
| 実装難易度 | 低（β1本＋channel減衰） | 中（b/w 2系統＋非対称WY因子） | デコードC実装の差は小（行列更新の係数形の差）。学習カーネルの差が主 |
| KV | 不要（固定状態） | 不要（固定状態） | 差なし。フル層KVは別文書（kv-cache-design.md） |
| 品質 | GDNより強いがβ束縛が制限 | 1.3B/100BでMamba-2・GDN・KDA・Mamba-3中超える総合最良、RULER multi-key retrievalで顕著 | ablation: 両ゲート寄与、**erase側が主** |

### P3採用決定

- **線形層はGated DeltaNet-2を維持（変更なし）。** KDAは縮退フォールバック（b/w束縛）として同一コードパスに内包可能であり、独立採用の理由がない。
- **デコードC実装の手本はkimi-k3-in-cの逐次再帰形**（`k3_kda_step`相当をGDN-2係数形で）。チャンク形は学習側に隔離（同リポジトリの分離判断に倣う）。
- **KDAの工学的詳細を借用:** short-conv frontend（k=3〜4、SiLU融合）、q/k L2Norm、sigmoid有界減衰（g_min=-5）、full-rank出力ゲート、A_logのhead-index規約。
- **比率4:1〜5:1を維持。** Kimi（≈2.9:1）・LFM2（3:1）より線形寄りだが、GDN-2の検索保持力を根拠に正当化。フル層側はkv-cache-design.mdの圧縮で補う。
- 非採用: CED、LFM2式「線形注意なし」選択（我々はGDN-2のRULER実績を優先）、SGLangコードの直接利用。

## 5. 参照URL

- https://github.com/FareedKhan-dev/kimi-k3-in-c
- https://arxiv.org/html/2605.22791
- https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention
- https://github.com/NVlabs/GatedDeltaNet-2
- https://github.com/sehaxe/burn-kda
- https://huggingface.co/LiquidAI/LFM2-24B-A2B
- https://www.liquid.ai/blog/lfm2-24b-a2b
- https://arxiv.org/html/2511.23404
- https://fromthematrix.dev/posts/local-llm-speed-benchmark-strix-halo/
- https://docs.sglang.io/docs/advanced_features/attention_backend
- https://github.com/sgl-project/sglang/blob/4a50cd78/sgl-kernel/csrc/cpu/mamba/fla.cpp
