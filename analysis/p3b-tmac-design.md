# P3b T-MAC設計（gate/up INT2・down INT4＋LUT逆量子化なしdot product）

- branch: `docs/p3/tmac-design`（本wtのみ。コミットしない）
- 性質：文書のみ。実装禁止（`.c`/`.h` 新規作成・既存改変なし）、重負荷実行なし（新規bench・DL・学習実行なし。既存計測値・コード読解のみ）。
- 前提：`analysis/p3-architecture.md`（P3a）§6・`analysis/p3-inference-engine.md` §3・`analysis/p3-training-scale.md` §1・`analysis/roofline.md` D4・D6・`knowledge/papers.md` §1を最優先前提とし、矛盾させない。矛盾発見時は本書を直さず記録し、P3本体への申送りは行わない（本書はP3bの一部として完結する）。
- 位置づけ：量子化副線の設計のみ。W8A8本線・MoE粒度・語彙head・GDN-2・KV・学習スケール・デコード投機の再定義はしない（必要分は依存として輸入する）。
- コスト表記：相対コストは Phase G Step 4 micro（`jt_gemm_mat_f32`、Mr=12×Nr=4）を「中」とした相対（小＜中＜大）。絶対人月・絶対期間は記さない（P3a §冒頭と同一）。
- 数値の使い分け：帯域上限は公称値を使わず保守値（N100 約1.3B・Ryzen 約0.4B）を上限目安に使う。byte/paramは素朴値（INT4 0.5B/p・INT2 0.25B/p・INT8 1.0B/p）とスケール込み値（Q4_K_M相当 約0.56B/p）を使い分けて明示する。

## 0. 境界条件と既存資産との関係（再定義ではなく境界）

- 対象は routed expert の低ビット path のみ：gate/up INT2・down INT4＋T-MAC LUT逆量子化なしdot product。共有expert（常時オン2つ）・注意射影・emb/head INT8は対象外であり、W8A8本線に据え置く。
- 使い分け（P3a §6・P3-inference §3と同一。再定義しない）：**W8A8 INT8経路を本線**（decode GEMVを含む確実な帯域半減＋整数SIMD加速）、**T-MAC LUT（INT2/INT4 routed）をGEMM体制（Mが大きいprefill・学習）の副線**に限定。decode GEMVではQLUT構築（act走査＋8×16列挙）がamortize不可のため本線にしない（roofline D4末尾・D6）。
- `src/tmac.c`（P1スカラー核＋P2集計ベクトル化）との関係：
  - 本設計は `src/tmac.c` の固定仮定をそのまま引き継ぐ：`g=4`固定・mirror consolidation（16→8、符号反転で復元、ロスレス）・`act_group=32`（連続32 activation＝8グループで1組のscale/bias共有）・weight scaleは単一`w_scale`に縮退（将来per-block化）・`bits∈{1,2,3,4}`（bits=3許可済み）・fast aggregation実装なし（NMSE 2.5倍劣化のためデフォルトOFF）。
  - 本設計は `jt_tmac_lut_ctor`（動的QLUT＋scales/biases生成）＋`jt_tmac_lookup_accum`（int8参照→int32累積→最後にscale/bias乗算1回/ブロック/プレーン）の2段構成を変えない。gate/upは`bits=2`、downは`bits=4`で同関数を使い分ける（カーネル書き分けなし。bit-serialの利点）。
  - SIMD方針も `src/tmac.c:36-51` のP2決定を維持する：LUT参照はスカラー維持（`jt_tmac_plane_acc`）、集計のみベクトル化（AVX2 `__m256d`／NEON `float64x2_t`、mul/add分離・FMA不使用・b昇順合算でbit同一）、AVX-512パスなし（ZMM downclock懸念・bits≦4は256bitに収まる・N100/i5-8th baseline移植性）。詳細は§4。
- `src/w8a8.c`（Stage 3 W8A8本線）との関係：
  - `src/w8a8.c`は対称ゼロ点0 INT8（`q∈[-127,127]`、`-128`不使用）・per-tensor／per-channel・int32蓄積（`K≦JT_BWD_MAX_WIDE`で`|acc|≦6.6e7<2^31`）・AVX2 emul exact／AVX512-VNNI dot限定／AVX-VNNI予約・FMA不使用・`M==0`起動スキップ・既定OFFフラグ（`jt_w8a8_set_enabled`）の本線である。
  - 本設計のT-MAC pathはこの本線を置き換えない。共有expert・注意射影のINT8 GEMMは `jt_w8a8_gemm_*` のまま。routed gate/up/downのみT-MAC副線に載せる。学習時のfake-quant／推論時の真カーネルの二経路構成（§3）はW8A8の同構成（`fakequant_*`＋`gemm_*`＋既定OFFフラグ）を踏襲するが、モジュール・フラグ・量子化粒度は共有しない（混線防止）。
  - 精度契約の違い：W8A8は整数exactのためSIMD路とスカラー路がbit同一（G3）。T-MACはQLUTの動的量子化（int8化誤差）を含むため、fp32基準とはbit同一にならない（G2/G4で判定。§5）。SIMD集計路とスカラー集計路の間だけはbit同一を要求する（§4・§5）。

## 1. ビット幅別テーブル構成（INT2/INT4のLUT形状・サイズ）

- 固定値（`include/jimotono/tmac.h`と同一。再定義しない）：
  - `JT_TMAC_G=4`、`JT_TMAC_LUT_HALF=8`、`JT_TMAC_LUT_FULL=16`、`JT_TMAC_ACT_GROUP=32`。
  - LUT粒度`g=4`は固定（`g≧5`はテーブル肥大で遅いため。papers §1・T-MAC原典）。
  - `act_group=32`はJIMOTONO仮定（原典は`g=4`で8値量子化のみ明示。`knowledge/papers.md` §1）。
- 形状（row-major、little-endian前提）：
  - `act`: `[N][K]` float（`N`=行数＝トークン行、`K`=内積次元）。
  - `qlut`: `[N][ngroups][8]` int8、`ngroups=K/4`。`qlut[g*8+i]`はパターン`p=8+i`（MSB=1、LSB-first：bit j ↔ a_j）の量子化値。`p<8`はミラー対`15-p`（格納index `7-p`）の符号反転で復元（`-128`の符号反転はint域→int32 acc加算で厳密）。
  - `scales/biases`: `[N][nblocks]` float各1配列、`nblocks=K/32`。block `bb`はgroups `[bb*8, bb*8+7]`（＝activation 32個）を共有。
  - `idx`: `[bits][ngroups]` uint8（1ドット分。下位4bitのみ使用`0..15`、上位nibble無視。将来のnibble分割2-lookup拡張用に予約）。
  - bit-serial線形変換：重みbit `0/1`を`s=-1/+1`に写像しfloat乗算を加減算のみに（LUT値域最小化＋量子化誤差削減。バイアスは最後に一括補正。papers §1核心5）。
- ビット幅の割付け（AGENTS.md §2.2維持）：
  - routed gate/up：INT2（`bits=2`）。2プレーン＋shift-add（`Σ_b 2^b * plane_b`、`b=0..1`）。
  - routed down：INT4（`bits=4`）。4プレーン＋shift-add（`b=0..3`）。
  - 計算量はbit数に比例（1-bit=1パス、4-bit=4パス）。カーネル書き分け不要（T-MACの最大利点。llama.cppの3-bit `8%3`地獄を回避）。
- サイズ式（`jt_tmac_qlut_bytes`／`jt_tmac_param_bytes`と同一）：
  - `qlut_bytes = N*(K/4)*8` byte。`param_bytes = N*(K/32)*sizeof(float)`（scales／biases各1配列分）。
  - 1ドット分の`idx`（unpacked、lookup直前形）：`bits*ngroups = bits*(K/4)` byte。
  - 格納形（SSD焼き、packed想定）：`bits*(K/4)` nibble＝`bits*K/8` byte／出力列。unpackedの1/2。nibble packはFP4 KVと同一思想（別ページ参照なし。§2）。
  - 注意（MINOR-2。`tmac.h:52`）：`qlut_bytes`単体は`K%4`、ctorは`K%32`を要求。`K=4`等では前者OKでも後者INVAL。`param_bytes`も`K%32`要求のため実害なし。本設計のexpert dims（例 `n=256`・`h=64`）はいずれも32の倍数のため影響なし。
- 例（`K=256`、gate/up入力側）：
  - `ngroups=64`、`nblocks=8`。
  - `qlut`/行＝`64*8=512B`、scales＋biases／行＝`8*4*2=64B`。
  - `idx`/ドット（unpacked）：INT2＝`2*64=128B`、INT4＝`4*64=256B`。
  - 格納（packed）：INT2＝`64B`/列、INT4＝`128B`/列。
  - 参照：`roofline.md` D4の参考値（bits=4・K=256で重み側1024B→idx 256Bの4×減）はunpacked形の値であり、packed格納ではさらに1/2になる。ただしGEMV体制ではQLUT構築（`src/tmac.c:230-385`）がamortizeされず支配的になるため、削減倍率をdecode本線の根拠に使わないこと（§0と同一）。
- QLUTの中身（`src/tmac.c:12-22`と同一）：
  - グループ内線形結合 `F[g][p] = Σ_j s_j(p)*a_j`、`s=+1`（bit=1）／`-1`（bit=0）、LSB-first。
  - ブロック（32 acts＝8 groups）毎に `max_abs = max|F|`、`scale = max_abs/127`、`bias = Σ 32 acts`。対称量子化のためmirror符号反転はint8レベルで厳密（端`-128`はint32で復元）。
  - `q = clamp(round(F/scale), -128, 127)`。`max_abs==0`は`scale=1/bias=0/q=0`。
  - 参照：plane寄与＝`(scale*acc + bias)/2`（0/1→±1逆変換）、`out = w_scale * Σ_b 2^b Σ_bb (scale[bb]*acc_bb_b + bias[bb])/2`。`acc`はint32（ブロック内8加算、範囲±1024でオーバーフロー不能）。

## 2. スケール配置（重みと同ページ方針・二重スケール）

- 2種のスケールを区別する（混同禁止。`knowledge/papers.md` §1 C実装2・JIMOTONO適用注意2）：
  - (a) weight scale：オフライン重み側。`group_size=128`がP1では単一`w_scale`に縮退（将来per-block化）。独自バイナリはbit-plane済みで焼く前提。
  - (b) LUT scale/bias：オンラインactivation側。`act_group=32`毎の動的量子化（`scales/biases[N][K/32]`）。SSDに焼かずDRAM/arenaに置く。
  - 最終乗算は1回だけ：int8参照→int32累積の後に`weight scale × LUT scale`を乗算（逆量子化回避。`tmac.c`・`tmac.h`と同一）。`scale-first`（累積前の事前乗算）と`final`の2方式のうち`final`のみ。事前乗算の混入は設計改訂。
- 重みと同ページ方針（本節の決定事項）：
  - weight scaleはそれが掛かるbit-planeと同一4Kページに格納する。別ページのメタデータ読みをdecode／prefill本線に混入させないこと（FP4 KVの「スケール同一キャッシュライン隣接」・PagedAttentionの「スケール別ページ禁止」と同一思想）。
  - 64Bライン整列：bit-plane済み重みはCPUキャッシュライン整列（64B `posix_memalign`／`_aligned_malloc`相当。`papers.md` §1 C実装1）。将来per-block化（`group_size=128`）時は、block先頭64Bライン内に当該blockのscaleを隣接配置し、走査順＝格納順（転置を発生させないこと）を維持する。
  - expertレコード単位：MoEではルーティング共起でディスク配置を並べ替え、1レコードを「expert丸ごと（INT4 down＋INT2 gate/up＋LUT用scale）の4K倍数」に（`knowledge/papers.md` §142と同一）。ホット／コールド分離とセット。レコード跨ぎのscale参照は禁止。
  - LUT scales/biasesは上記SSDページ方針の対象外：オンライン生成・DRAM常駐（`jt_arena_alloc`推奨・64B整列推奨。スカラーP1核は非整列でも動作、将来SIMDで必須の予約は維持）。SSD永続・prefix共有の対象にしないこと（SWA FP8局所と同様、非永続）。
- 二重スケールの精度・速度トレードオフ（未解決として記録。断定しない）：
  - `weight block scale × LUT scale`の積をどこで当てるか（§1参照式の`w_scale * Σ ...`の位置）は精度・速度が変わるが、本設計では最終1回乗算に固定する。変更は設計改訂。
  - Q4_K_M相当のスケール込み換算（約0.56B/p）と素朴0.5B/pの使い分けはbyte/token見積り（§6）で明示すること（P3-inference §3未解決と同一）。
- fail-closed（`src/tmac.c`・`src/w8a8.c`と同一流儀）：
  - 全公開API相当は書込み前に検証を完了し、失敗時は出力を更新しない。非有限（NaN/Inf）入力・非正scale・`q`値域外は拒否（`errno=EINVAL/ERANGE`＋`goto cleanup`相当）。`size_t`積はオーバーフロー番兵（`arena.c`の`jt_align_up`方式）。

## 3. 学習時と推論時の切替（STE／量子化誤差）

- 決定事項：学習時はfake-quant＋既存fp32 GEMM核、推論時は真のT-MAC LUT（ctor＋lookup）の二経路とし、明示フラグで切り替える。自動切替・黙示の精度切替は禁止。
  - 学習時（fake-quant）：`quant→dequant`後にfp32 GEMM（`jt_gemm_mat_f32`相当の既存核を再利用。新規GEMM実装なし）。量子化ノイズのみを付加し、STE（Straight-Through Estimator）で逆伝播する：順伝播は量子化値、逆伝播は量子化を素通し（`dL/dw ≈ dL/dq`）とする。top-kのhard選択マスク・dropマスク・renormalize分母`S`経由の2次項はstraight-throughとして無視する（`analysis/gemm-design.md` §3.3と同一仕様。ハイパーパラメータではなく仕様として固定。変更時は設計改訂）。
  - 推論時（真のLUT）：オンラインで`jt_tmac_lut_ctor`相当の動的QLUT＋scales/biasesを生成し、`jt_tmac_lookup_accum`相当のint8参照→int32累積→最終scale乗算でdot productを求める。逆量子化（事前dequant＋fp32 GEMM）を本線に混入させないこと。
  - フラグ：W8A8の`jt_w8a8_set_enabled`（既定OFF＝fp32）と同形の明示フラグを想定する（名称・APIの新設は実装であり本書では行わない。切替契約のみを定義する）。既定はfp32のまま。ON時のみ低ビット経路を使う。OFF時は呼出し側が従来通りfp32核を呼ぶ（速度不変）。
- 量子化誤差の扱い（事前定義。測定で置き換えない）：
  - fake-quantが模す誤差はtable quantization誤差（int8 QLUT＋scale/bias。`src/tmac.c` ctor式：`F`列挙→`maxabs/127`→`round`→clamp）のみ。fast aggregation誤差は含めない（実装しないため。§0）。
  - gate/up INT2とdown INT4の誤差は分離して記録する（FRI-MxMoEの「エキスパート内サブレイヤーの異なるロバスト性」の捉え方としての参照に限定。`https://aclanthology.org/2026.acl-long.982/` 。プロファイリング高速化の輸入はP3b以降の判断であり本書では行わない）。
  - W8A8 INT8 fake-quant（`jt_w8a8_fakequant_*`）とは別物であり、INT8本線の誤差に低ビット誤差を上乗せ計上しないこと。測定時はINT8-only対照を併走させ、INT2/INT4分の増分を分離すること（§5）。
- HGQ-LUTとの区別（`knowledge/papers.md` §1と同一）：
  - T-MACはCPU動的LUT（重みbit-plane分割＋タイル順permutation＋interleaveをオフライン、QLUT＋scales/biasesをオンライン動的生成）。学習は通常のQAT／GPTQ／BitNetでよく、推論時コンパイル不要。
  - HGQ（FPGA用、BN＋Dense＋Actを真理値表に静的展開）とは目的も実装も異なる。AGENTS.md §3.3の「HGQ-LUT方式」表記はT-MAC文脈では使わない（P3本体判断。`analysis/p3-inference-engine.md` §3と同一。本書では表記変更しない）。
  - オフライン焼き（bit-plane＋permute＋interleave）のみが推論前処理であり、学習→推論の重み変換に追加の学習実行を要しないこと。

## 4. AVX2／AVX-VNNI／AVX512-VNNIでのLUT参照実装方針

- 基本方針（`src/tmac.c:36-51`のP2決定を維持。再定義しない）：
  - LUT参照（gather的8 lookup）はスカラー維持（`jt_tmac_plane_acc`）、集計（`(sc*acc+bi)*0.5*2^b`のブロック内bits方向、最大4）のみベクトル化する。shuffle／gatherの完全ベクトル化は行わない。
  - 理由：int8 8要素のshuffle化はlane複製・mirror符号復元分岐が増え、8要素ではペイしないため。16B＝NEON 128bit一致は将来のlookup完全ベクトル化の予約位置に留める。
- AVX2（本線のSIMD）：
  - 集計のみ`__m256d`（4xdouble）1本で1ブロック分を一括計算（`jt_tmac_block_sum_avx2`と同形）。要素毎の演算順序はスカラーと同一（mul→add→mul(0.5)→mul(pow2)）、レーン合算のみ`b=0..bits-1`のスカラー順序で加算してbit同一性を保つ。FMA（`fmadd`）は使わない：1丸め化でスカラー（mul→add 2丸め）とbit同一にならないため。mul/add分離でIEEE丸め順序を保存する。
  - 将来のlookup完全ベクトル化（`_mm256_shuffle_epi8`）は予約に留める（実装しない）。`#ifdef __AVX2__`ガード。アライメント非依存（非整列load/storeのみ。`_mm256_storeu`相当）。64B整列は推奨のまま必須化しない。TU毎opt-in（global強制不可。`CMakeLists`方針と同一）。
- AVX-VNNI（N100用）：
  - T-MAC LUT pathには追加しない。`src/w8a8.c:241-250`の調査結果の通り、`_mm256_dpbssd_epi32`（s8×s8）は`-mavxvnniint8`（`__AVXVNNIINT8__`）でのW8A8 dot用であり、LUTのshuffle参照を加速しない。本設計のT-MAC副線はAVX2路にフォールスルーする（実装任意のため予約のみ。非対応CPUでは同一k順スカラーフォールバックでbit同一）。
  - VNNIの恩恵はW8A8本線（decode GEMVを含む）に計上し、T-MAC副線の効果に上乗せしないこと。
- AVX512-VNNI（Ryzen AI 360用）：
  - T-MAC LUT pathは設けない（`src/tmac.c:40-51`の見送り理由と同一）。ZMM使用で最大30%ダウンクロック（DESIGN.md §2.3）・bits≦4は256bit（4xdouble）に丁度収まり512bitの追加スループットなし・LUT参照がgather律速で演算律速でない以上ZMMの利益なし・N100/i5-8th baseline移植性を優先。一般GEMMはAVX2路を共用し、未検証コードを最小化する。将来演算律速部が出れば再検討（再検討自体は設計改訂）。
  - `clang`に`__AVX512__`マクロは存在しないため、仮に将来設ける場合は`defined(__AVX512F__) && defined(__AVX512VNNI__)`でガードすること（`src/w8a8.c`と同一注意）。
- ARM NEON（移植予約）：
  - 集計のみ`float64x2_t`で2要素ずつ（`jt_tmac_block_sum_neon`と同形）。要素内順序はスカラー同一、合算はb昇順。FMA（`vfmaq_f64`）不使用、mul/add分離。`vst1q`のみ・gather部スカラーのため非整列可。
  - lookup完全ベクトル化（`vqtbl1q_u8`）は予約に留める（`g=4,int8`で16B＝NEON 128bitに丁度の位置づけは維持。`knowledge/papers.md` §1 C実装1）。
- 共通契約：
  - C11・`restrict`・`errno`＋`goto cleanup`・リトルエンディアン前提（`common.h`でビッグエンディアンをコンパイルエラー）・クロスプラット（Linux／macOS／Windows同一ソース）。
  - SIMD集計路とスカラー集計路はbit同一であること（G3。§5）。整数`acc`は加算順序不変で完全一致するため、float化の順序を既存と同一に保てば出力はbit同一になる（W8A8 G3と同一考え方）。

## 5. G2／G4ゲート条件（精度基準を事前定義。測定で置き換えない）

- 用語（`analysis/gemm-design.md` §5.4–§5.6と同一定義を輸入。再定義しない）：
  - G1＝bit一致（関数変更なし・順序のみ）。G2＝関数変更（統計的同等性を超える関数等価性）。G3＝同一関数内最適化（bit一致 or 相対差1e-6）。G4＝数値等価だがrouting結合で関数的に発散し得る最適化（同一perm／drop／renormalize仕様の下でGEMM内丸め順序のみが変わり、重み→routing帰還で長時間軌道が発散し得るもの）。
  - drop率・renormalize発火率はG2/G4合否に使わず、§1.2トリガ（steady-state判定）相当で分離記録する（gemm-design §1.2・§5.4と同一）。
- G2-quant（関数変更：fp32 → INT2/INT4 fake-quant。事前定義）：
  - 対象：routed gate/up INT2 fake-quant＋routed down INT4 fake-quant（共有expert・注意射影はINT8維持。emb/headはINT8＋factorized維持）。INT8-only fake-quant対照を併走させ、低ビット分の増分を分離すること。
  - 条件：同一シード・同一データ順・同一スレッド数・同一スレッド配置。TinyStories短走（200 steps以下。1000 steps禁止。gemm-design §1.2 G2改訂と同一）・逐次・単一プロセス・実行前`uptime`確認。
  - 合否（4項目。train lossはいずれにも使わない。TinyStories実言語CEのvalのみ。gemm-design §5.6と同一）：
    1. 最終val差 ≦1%（相対差。合成回帰のtrain飽和後の相対差は不使用）。
    2. 収束単調性：後半50%のval相対差推移を確認。単調増加ならdynamics変化、安定なら過渡応答と記録する（許容判断はしない。推移の事実のみ記録）。
    3. 勾配ノルム分布：平均 ±10%以内（分散は記録のみ。合否に使わない）。
    4. INT8-only対照との差分記録：INT8-only→INT2/INT4化の増分val差を記録し、低ビット起因とINT8起因を分離すること（分離不能時はG2通過としない）。
  - 不合格項目があれば原因分析を行う（許容判断禁止）。
- G4-trueLUT（fake-quant → 真のT-MAC LUT。事前定義）：
  - 対象：同一quant仕様（同一perm／drop／renormalize仕様・同一scale粒度・同一bits）の下でのfake-quant（fp32 GEMM）から真のLUT（ctor＋lookup_accum）への切替。丸め順序のみが変わり、重み→routing帰還結合で長時間軌道が発散し得るためG4とする。G3 strict（stepwise 1e-6）はdrop=0域・単位レベル（テール・単体ドット・短期再走行）に適用を限定し、drop>0の長時間系統軌道には適用しない（適用すれば仕様内丸めが必発で不通過となるため。gemm-design §5.6と同一）。
  - 合否（4項目。train loss不使用。gemm-design §5.6と同一）：
    1. 構造的等価性：perm／off／dropマスクbit一致、gate初期一致（初step drop数の一致）、テールbit一致、決定論性（同一条件再走行のstderr bit一致）。
    2. 統計的同等性：最終val差 ≦1%（相対差。G2と同一閾値を流用）。
    3. 増幅有界性：val差軌道の形状で判定。飽和（<0.3%に収束・単調増加なし）＝完全通過、線形（~0.8%への漸増）＝条件付き通過、指数（数%超への発散）＝不通過。
    4. 性能目標：GEMM体制でのQLUT amortize（§6）を維持すること。decode GEMVへのLUT混入は不通過（性能以前の設計違反）。
  - SIMD集計（AVX2／NEON）vs スカラー集計はG4ではなくG3（bit一致）を要求する（§4）。不一致は実装バグとして分離し、G4系統軌道の評価に進まないこと。
- G3（同一関数内最適化。事前定義）：
  - 対象：Step 4相当のマイクロ最適化（packing・タイル・スレッド分割等の順序不変域）。判定はbit一致または相対差1e-6（gemm-design §5.2を維持）。G2基準（1%）は流用しない。
  - FMA導入時は別途「FMA vs mul/add分離」の差分評価を要し、本書の基準をそのまま流用しないこと（契約積算順序が変わるため。gemm-design §5.3と同一）。

## 6. 容量・NVMe・起動回数との整合（受入条件。効果の上乗せ禁止）

- 容量（dense≦2GB INT4／全体≦8GB。AGENTS.md §8）：
  - 換算は素朴値とスケール込み値を使い分ける：INT2 0.25B/p・INT4 0.5B/p（スケール・メタ別）／Q4_K_M相当なら約0.56B/p（`analysis/p3-architecture.md` §4・`analysis/active-upper-bound.md` §1と同一）。
  - active 100M基準の内訳目安はP3a §4・P3-inference §5と同一（100M INT4≒50MB＋縮小head INT8約12.7MB＋KV微小≒63〜100MB/token）。低ビット化の容量利得をW8A8本線の帯域半減に二重計上しないこと。
- NVMe（1トークンあたり≦103MiB。AGENTS.md §8）：
  - T-MAC副線の格納は§2のpacked形（INT2 0.25B/p相当・INT4 0.5B/p相当＋同ページscale）で見積もること。QLUT（DRAM構築）・LUT scales/biases（DRAM常駐）のSSD往復を計上しないこと（往復させない設計のため）。
  - 想定約100MB/tokenは境界上のため、pin-hit率（StickyMoE熱pin相当）＋Delta持続率で103MiB内に収めることをP3受入条件とする（P3-inference §5と同一）。論文値転載で代替不可。
- カーネル起動（50回/token以下。AGENTS.md §8）：
  - QLUT ctorを1カーネルとして計上すること（行列毎の再構築を1起動に融合する前提）。decode GEMVでのQLUT再構築は起動回数を悪化させるため、本線化しない根拠の一つとする（§0と同一）。
- 推論速度（N100 5tok/s・Ryzen 30tok/s）：
  - 本書で速度値の断定・Phase G効果への上乗せ計上は行わない（P3-inference §0と同一）。受入はすべてP3推論実装時の実測をゲート条件とする。

## 参照URL一覧

- T-MAC原典（必読）：https://ar5iv.labs.arxiv.org/html/2407.00088
- T-MAC arXiv：https://arxiv.org/abs/2407.00088
- T-MAC実装：https://github.com/microsoft/T-MAC
- BitNet実装：https://github.com/microsoft/BitNet
- 混合精度参照（INT2/INT4割付けの参照）：https://huggingface.co/OsaurusAI/ZAYA1-8B-JANGTQ_K
- FRI-MxMoE（サブレイヤー別ロバスト性の捉え方のみ）：https://aclanthology.org/2026.acl-long.982/
- W8A8 INT8本線（SGLang CPU）：https://docs.sglang.io/docs/hardware-platforms/cpu_server

## 7. Stage 3b-2実測記録（条件付き通過）

- TinyStories T=512・200/500 steps（逐次・単一）：fp32基準 2.9470/2.4137、
  INT8対照 +0.21%、fake-quant +1.21%/+1.16%、真LUT +1.30%/+0.98%。
- 判定：真LUT（本番経路）が500 stepsで+0.98% → 通過。fake-quant +1.16%は
  marginalだが本番経路でないため許容。INT8対照+0.21%はW8A8本線の裏付け。
- 真LUT vs fake-quantの差+0.18%（500 steps）を明記。真LUTは収束中。
- proxy解像度仮説：d=64 vs d=1024の量子化感度差。大dほど鈍感（冗長性）の
  一般則を参照。S1でdesign scale実測を必須とする。

## 禁止事項の遵守記録

- 他wt接触・checkout/switch：なし（本wt・本branchのみ。`git rev-parse --show-toplevel`＋`git branch --show-current`で確認）。
- 実装：なし（`.c`/`.h` 新規作成・改変なし。`src/tmac.c`・`src/w8a8.c`・`include/jimotono/tmac.h`・`include/jimotono/w8a8.h`は読解のみ）。
- 重負荷実行：なし（新規bench・DL・学習実行なし。既存計測値・コード読解のみ）。
- コミット：しない（作業ツリー内文書のみ）。
- 4文書混交：なし（MoE粒度・語彙・学習スケール・デコード投機の再定義をせず依存輸入に留めた。量子化副線のみ）。
