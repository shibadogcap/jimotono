# knowledge/papers.md — Phase 0 文献調査メモ (初期版)

> 作成: 2026-09-13 / Phase 0 完了条件「T-MAC / GDN-2 / StickyMoE / NeuroPrefetcher 読解」の成果物。
> 運用: ROADMAP 9により週次更新。更新は docs/ ブランチで。

---

## 1. T-MAC: LUT-based mpGEMM (2407.00088)

### 核心 (5行)
- **bit-serial分解**: `A×W = Σ 2^i·A×Wi`。n-bit重みをn枚の1-bit面に分解し、bit幅に線形スケール。W4/W3/W2/W1でカーネルを書き分けない。
- **LUT構築**: activation `[1,g]` × 全bitパターン `[g,2^g]` をオンライン事前計算し `[1,2^g]` テーブル化。例 `g=4` → 16エントリ。重みのg-bitがそのままインデックス。
- **テーブル圧縮**: 素朴 `2^g` (g=4→16) を mirror consolidation で `2^(g-1)` (符号反転で復元、ロスレス) + table quantization (fp16→int8+scale/bias) で合計1/4に。
- **bit幅の違い**: 計算量はbit数に比例 (1-bit=1パス、4-bit=4パス+shift-add)。llama.cppの3-bit `8%3` アライメント地獄を回避できるのが最大利点。
- **bit-serial線形変換**: `0/1` を `s0=-1,s1=+1` に写像しfloat乗算を加減算のみに。LUT値域を最小化し量子化誤差削減。バイアスは最後に一括補正。

### 学習→推論の流れ (注意: HGQ-LUTとは別物)
- T-MACは **CPU動的LUT**: (1) オフラインで重みをbit-plane分割+タイル順permutation+interleave、(2) オンラインで activation から `QLUT + LUT_Scales/Biases` を動的量子化で生成 (`act_group=32/64`)、(3) `TBL/PSHUF` lookup + int32累積 → 最後にweight scale×LUT scaleを乗算。
- 学習は通常のQAT/GPTQ/BitNetでよく、推論時コンパイル不要。HGQ (FPGA用、BN+Dense+Actを真理値表に静的展開) とは目的も実装も異なる。JIMOTONO文書の「HGQ-LUT方式」という表現はT-MAC文脈では使わない (AGENTS.MD 3.3の修正が必要。P1着手前にAGENTS側を「T-MAC動的LUT」に修正すること)。

### C実装への示唆
1. **レジスタ常駐 + 64B整列 + K-first**: LUTはL1でなくレジスタに (`NEON:vqtbl1q_u8`, `AVX2:_mm256_shuffle_epi8`)。`g=4,int8`で16B=NEON 128bitに丁度。workspaceは `posix_memalign(64)` / `_aligned_malloc(64)`。tilingは `Mt_m` を大きくし同一LUT再利用最大化。
2. **group size 3種を区別**: `g=4` (LUT粒度、5以上は遅いので固定推奨)、`group_size=128` (weight scale粒度)、`act_group_size=32/64` (LUT動的量子化粒度、JIMOTONO仮定。T-MAC原典は `g=4で8値量子化` のみ明示)。
3. **逆量子化回避**: (a) オフラインbit-plane+permuteでunpack消去、(b) LUT値int8化・累積int32・スケール乗算は最後に1回だけ、(c) nibble分割lookup案 (`&0x0F` と `>>4` で1命令2 lookup。JIMOTONO独自提案、原典記述ではない)。fast 8-bit集約 (`avg/rhadd`) は精度劣化 (NMSE 2.5倍) のためデフォルトOFF。
4. **C11注意**: `restrict` + `_Alignas(64)`、ホットパスは `#ifdef __AVX512__ / __AVX2__ / __ARM_NEON` 分岐、`errno + goto cleanup`。

### 性能数値
- BitNet-b1.58-3Bで `30 tok/s (1コア), 71 tok/s (8コア) on M2-Ultra, 11 tok/s on RPi5`。llama.cpp比 最大4x、省エネ70%。
- Llama-2-7B/13B GEMVで `11.2x/5.8x/4.7x/3.1x (1/2/3/4bit)`。llama.cppは4→2bitで高速化せず3bitで15%悪化するがT-MACは線形改善。
- 電力: M2-Ultraでエネ -20.6%/61.2%/51.3% (W4/W2/BitNet)。精度はtable quantizationのみならllama.cppと同等。

### JIMOTONOへの適用注意
1. **64x64正方形タイルとLUT再利用の衝突**: T-MACは `Mt_m` 大・`Kt_k` 小が前提。64x64固定では再利用率が落ちる。K-first走査維持 + `Mt` 方向にexpert並列で束ねるかタイル形状再検討。
2. **混合精度 (shared INT8 / down INT4 / gate-up INT2) と二重スケール**: weight block scale × LUT scaleの積を scale-first/final どちらで当てるかで精度・速度が変わる。独自バイナリは最初からbit-plane済みでSSDに焼く。
3. **SIMD分岐**: AVX2 (lane複製要) / AVX-512 / NEON でlookup命令が異なる。`#ifdef` 分岐 + 64B整列必須。

### 参照
- https://ar5iv.labs.arxiv.org/html/2407.00088 / https://arxiv.org/abs/2407.00088
- https://github.com/microsoft/T-MAC / https://github.com/microsoft/BitNet

---

## 2. Gated DeltaNet-2 (NVIDIA Research 2026-05)

### 核心: erase / write 分離 (5行)
- Gated DeltaNet / KDAは単一スカラー `β_t` で「古い連想の消去」と「新値の書込み」を同時制御していた。
- GDN-2はこれを `b_t` (key側・消去) と `w_t` (value側・書込み) のチャネル単位ベクトルに分離。
- KDA継承のチャネル単位decay `D_t=Diag(α_t)` は大域忘却、eraseは選択的修正、writeは選択的挿入を担う。
- `b_t=w_t=β_t·1` でKDAに、αもスカラー化でGated DeltaNetに厳密帰着。Ablationではerase側の寄与大。

### 固定サイズ running state・KV不要理由
```text
S_t = (I - k_t (b_t⊙k_t)^T) D_t S_{t-1} + k_t (w_t⊙v_t)^T
o_t = S_t^T q_t
```
- `S_t ∈ R^{dk×dv}` (論文はH=16, dk=dv=128)。デコードは `S_t` のみ保持し `O(1)` 更新。Softmaxのような `O(L)` KVキャッシュ不要。

### チャンクワイズ並列 (WY型)
- チャンク長: 論文固定 `C=64`。decay正規化を吸収すると純粋非対称delta漸化式になる。
- Intra-chunk: 64×64下三角solve (前進代入) + 密行列積。Inter-chunk: チャンク間のみ漸化式。
- C実装注意: (a) log-decay・累積和・stateはfp32でカーネル外計算、(b) q/kはL2正規化必須、(c) デコードはチャンク核を使わずトークン逐次核、可変長はcuSeqlens境界でstateリセット。

### 性能数値 (1.3B / FineWeb-Edu 100B, 系列長4K)
- 言語モデリング: Mamba-2/GDN/KDA/Mamba-3を上回り最良 (Wiki ppl 15.90、平均acc 53.11)。Hybrid (+SWA 2K) でも最良。
- RULER NIAH: S-NIAH-2@4K 93.0、S-NIAH-3@2K 89.8、MK-NIAH-1@4K 37.8で最良。干渉の強いmulti-keyで差が最大。
- 学習スループットH100で38.0→36.1Kt/sとflat、KDA比小オーバーヘッドのみ。

### JIMOTONOへの適用注意 (線形:フル=4:1〜7:1, 1C1T)
1. **デコードは逐次核・プリフィルのみチャンク化**: 1C1Tでは前進代入 (C=64逐次依存) がFMA 4チェーンを埋められない。デコードは rank-1更新でhead/dv方向に4分割しlatency隠蔽。チャンク核はプリフィル用にC=16/32縮小+L1/L2常駐優先。
2. **ゲート削減はwrite側から**: eraseチャネル性が効果大。INT2/INT4化でも `b_t` の精度を落とさず `w_t` のスカラー化・量子化を先に検討。
3. **Hybrid比率とstate配置**: SWA (JIMOTONOは128+CSA) が局所を担いGDN-2は長距離専用化できる。`S` は層あたり1MB (fp32) でL2/L3に載るサイズに維持。

### 参照
- https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention
- https://arxiv.org/abs/2605.22791 / https://github.com/NVlabs/GatedDeltaNet-2

---

## 3. StickyMoE soft-hard variant (2607.08780v1)

### Soft / Hard (5行)
- Soft: `L_cons = 1/(T-1) Σ ||g_t - g_{t-1}||_2^2`、全層平均。全損失 `L = L_CE + λL_cons + μL_bal (+ αL_hard)`、`μ=0.01`固定。
- 直感: 隣接トークンのgate分布差にL2ペナルティ。`Wr`直接＋`h_t`経由で表現も同時矯正する点がpost-hocと決定的に違う。
- Hard: 系列を非重複W窓分割。窓先頭を固定アンカーに `L_hard = (t-s(t))/W · ||g_t - g_{s(t)}||^2`。線形ランプで窓末ほど拘束強化。
- 直感: VQ-VAE commitment由来の半径拘束。Softだけの漸進ドリフト (小刻みに遠方へ移動) を防止。
- 併用 `λ=0.1, α=0.05〜1.0, W=4` が最良。

### fine-grained smoothness と coarse-grained commitment / Wの選び方
- Soft=ステップ幅 (微視的滑らかさ)、Hard=半径 (窓内全トークンが同一常駐expertで捌ける巨視的拘束)。片方だけでは「小刻みドリフト」「窓内ジッタ」が残る。
- 論文値: Soft-Hardは `W=4` 固定。Hard単体は `W=2` が最良、`W=8` は逆に悪化。初期値は `W=2〜4` から開始し (W=8はPareto確認用に留める)、SR/CHRでPareto選択。
- W選び方針: (a) LRU容量Cと対応 (C=2評価でW=4が最良)、(b) 意味的コヒーレンスspanに合わせる、(c) 大きすぎるWはPPL悪化。

### C実装・学習実装への示唆
- **学習**: C学習カーネルに `L_cons+L_hard` を追加するだけ。gate差分二乗で足り計算量無視可。`λ=0.05〜0.1` 開始、共有expert 2つは損失対象外 (常駐のため)。
- **推論×プリフェッチ案1 (アンカー固定先読み)**: 窓先頭で集合確定→窓末まで同一集合を `io_uring+O_DIRECT` でピン留め。`S∪L∪M` 和集合先読み量を窓内ゼロに。
- **案2 (Router-lookahead増幅)**: Stickyで `h_t` 自体が滑らかになるため予測器精度・Delta持続率がさらに上がる。W窓=予測ホライズンに一致させると効果最大。
- **案3 (境界-aware緩和)**: 文・関数境界ではペナルティをマスク。無差別ペナルティは `λ≥0.2` でPPL悪化要因。

### 性能数値
- SR削減59%: Soft `λ=0.5` で 0.71→0.30 (条件: top-1・C=2・4 experts・WikiText-2・8.8M/22M小規模)。miss削減3.92倍: CHR 0.54→0.88 (同条件)。JIMOTONO (数千expert×top-8〜16) への直置きは過大評価のため測定定義の再定義が必須。
- 品質: 低λは正則化として改善 (`λ=0.05` でPPL -4.1%)。`λ≥0.2` でトレードオフ顕在。
- 崩壊なし: 利用エントロピー全条件1.92bit以上。post-hoc router微調整は無効 (<0.5%)。
- 層別ではL1が最大改善 (85%減)、L0は抵抗大 (embedding未混合のため)。

### JIMOTONOへの適用注意 (細粒度2〜4M×数千、top-8〜16、共有2)
1. **top-1評価とtop-8〜16の乖離**: 論文SR/CHRはtop-1・C=2定義。数千expert×top-8〜16では集合一致で再定義し、W・Cをtop-k倍率で拡大 (W=8〜16) しないと過大評価。
2. **L0障壁**: L0は改善鈍化。層別λ (L0小、L1/L2大) ＋境界マスク必須。
3. **decodeでは層間一貫性に注意**: 層別expert構成では `L_cross` は無効。decodeはアンカー窓ピン留め＋投機プリフェッチで稼ぐ。

### 参照
- https://arxiv.org/abs/2607.08780v1 / https://arxiv.org/html/2607.08780v1
- https://github.com/alikayyam/sticky_moe.git

---

## 4. NeuroPrefetcher (ICPP 2026)

### 核心 (5行)
- レイヤ0をdense実行し特徴5点 (埋め込み/P/pre-attention sketch/post-attentionスカラー/g0) を抽出、トークン毎1回だけ共有予測器を実行。
- 予測器は特徴別proj→T=512埋め込み→4-block残差trunk→層別gate/upヘッドで全sparse層のセントロイドIDを一括出力 (206.8M params、モデルの2.86%)。
- IDをcentroid bank (K=512) でマスク展開し `gate & up` のintersectionを活性集合とする。正解セントロイド選択率~99%。
- ランタイムは活性集合と常駐集合をdiffしΔ行のみNVMeからfetch。dense prefix計算中にI/Oをオーバーラップ。
- オフライン準備: percentile閾値でsparse化 (再学習なし) → K=512にクラスタリング。

### Delta Prefetching と Router-lookahead
- **Delta**: sparse行の82〜85%が前トークンと同一。残り15〜18%のincoming行のみ読み込み。neuron-major単一ファイル + `find_runs_gap1` で `(offset,length)` 結合しランダムI/O削減。初トークンは全層一括、2トークン目以降deltaのみ。
- **Router-lookahead 71.6%** (Colibri `PILOT=1` 実測、GLM-5.2固有測定のためJIMOTONO MoEでは再測定要): 層Lのpost-attention状態に層L+1のrouterを適用すると真のtop-8を71.6% recall (ベースライン41.3%)。専用I/Oスレッドが次層expertを先読み。
- 両者は相補: NeuroPrefetcher型=トークン間delta (時間局所性)、Router-lookahead=層間先読み (空間局所性)。

### io_uring + O_DIRECT 要点 / Colibri式との使い分け
- `liburing` sliding-window: `batch=256`、ring sizeは2の冪、`prep_read→submit→wait_cqe+peekでdrain→resubmit`。GIL解放しdense計算と並列。(未確認: `engine/fast_pread_uring.c` のコード確認が必要。`find_runs_gap1`、`初トークン全層一括` も同様に要確認)
- `O_DIRECT` 前提: `posix_memalign(4096)` バッファ、short-readは `pread` リトライ。
- 判断基準: ホット (共有expert/attention/頻出routed) はLRU+pin+buffered、コールドdelta streamingはO_DIRECT。macOSに `O_DIRECT` なし→ `F_NOCACHE`、Windowsは `FILE_FLAG_NO_BUFFERING` のため抽象化層でbuffered fallback必須。

### 性能数値
- llama.cpp比 **7.9〜12.0倍** (model-exceeds-memory帯)。major page fault **11453/s→0.21/s**、CPU iowait **77%→4%**。
- トークン当りNVMe読み **103MiB** (条件: Mistral-7B FP16・62% sparsity・14.8GiB時。JIMOTONO目標化は条件付き)→ 9.0GiB時1GiB弱。遅延の80〜87%がNVMe I/O。
- 品質: 予測正解率~99% (centroid選択率。活性予測≠e2e正解率)、dense精度の92〜96%保持。

### JIMOTONOへの適用注意
1. **I/O抽象化は機能差を吸収**: C11層は `jt_io_pread_batch(fd,specs,dsts)` のspec配列I/Fに統一。Linuxのみio_uring+O_DIRECT、macOSは `F_NOCACHE`+`preadv`、Windowsは `OVERLAPPED`+buffered fallback。起動時 `iobench` 相当で実効帯域を測り較正。
2. **共起クラスタリングなしにdeltaは効かない**: MoEではルーティング共起でディスク配置を並べ替え、1レコードを expert丸ごと (INT4 down + INT2 gate/up + LUT) の4K倍数に。ホット/コールド分離とセットで設計。
3. **品質ノブとメモリノブを分離**: `d` (dense prefix長)=品質ノブ、`K_max`/sparsity=メモリノブ。`d*` 超過でwhole-layer読みに落ちるため起動時auto-budgetでcap決定。

### 参照
- https://github.com/nobeldhar/NeuroPrefetcher
- https://arxiv.org/abs/2608.22643 / https://doi.org/10.1145/3832810.3832862
- https://github.com/Scottcjn/colibri

---

## 5. Phase 1への申し送り (共通)

> 注意: `103MiB / 99% / 59% / 3.92x` のJIMOTONO目標化は条件付き (上記各章の条件を参照)。DESIGN.MD 6章の無条件目標化は過大評価のため、測定定義 (モデル・量子化・sparsity・C・top-k) を付記して評価すること。

- [ ] T-MAC: `g=4` 固定、fast aggregation OFF、独自バイナリはbit-plane済みで焼く。64x64タイル形状は再検討要 (Mt大/Kt小のsweep、expert並列束ねの実験計画をPhase 1で策定)。
- [ ] GDN-2: デコード逐次核 + プリフィル C=16/32、state fp32維持、q/k L2正規化、`b_t` 精度優先。
- [ ] StickyMoE: `λ=0.05〜0.1, W=2〜4` 開始 (W=8はPareto確認用)、共有expertは損失対象外、層別λ＋境界マスク。
- [ ] NeuroPrefetcher: `jt_io_pread_batch` 抽象化、共起クラスタリング＋4K倍数レコード、ホット=LRU/コールド=O_DIRECT。

## 6. llm.c / GPT-2 forward-backward (Phase 0完了条件の補完)

> ROADMAP 10の完了条件「llm.cをビルドし、GPT-2のforward/backwardを理解」の証跡。P0時点ではビルド実行なし・読解メモのみ。Phase 2 (学習エンジン) 着手前に実ビルドで検証すること。

- **対象**: karpathy/llm.c (GPT-2 forward/backwardのC/CUDA参照実装)。JIMOTONO学習エンジン (Phase 2) の出発点。
- **要点**: (a) forwardは layernorm→matmul→gelu→residual の融合単位で理解する、(b) backwardは dX先行・dW後回しのオーバーラップ (DESIGN 4.1) が前提、(c) Adam状態の8bit化・勾配チェックポインティングはPhase 2で実装。
- **未実施**: llm.cの実ビルド・GPT-2 tinyでの数値再現。Phase 2開始条件とする。
- 参照: https://github.com/karpathy/llm.c
