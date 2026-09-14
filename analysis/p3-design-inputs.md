# 調査項目4：P3統合設計入力 — active上限再計算・head縮小・目標再設定

- branch: `docs/p3/survey-train-design`（本wtのみ。コミットしない）
- 性質：調査＋文書のみ。実装禁止（語彙スパース化の実装はP4記録のみ）、重負荷実行なし。
- 前提データ（タスク受領値）：LFM2-24B-A2B（active 2.3B・112 tok/s）、Gemma-4 26B MoE（active 3.8B・124 tok/s）、
  語彙48,588（head約50M）。帯域前提：N100 約40GB/s、Ryzen AI 360 約80GB/s。
- 方法論の宣言：**toks/sのcross-vocab直接比較は行わない**（タスク禁止事項）。
  LFM2/Gemma値は「同語彙で並べる」のではなく「帯域→byte/token→実効効率η」の較正材料としてのみ使う。
  比較はすべてGB/sとbyte/tokenに正規化する。

## 結論（先に）

1. N100（5 tok/s）・Ryzen AI 360（30 tok/s）のいずれも、**帯域はactive 100M目標の律速ではない**。
   実効帯域の50%を見込んでも上限はN100でINT4換算約8B、Ryzenで約2.6B。律速はhead・固定費・SSD・起動回数。
2. 未縮小head（1024×48,588≒49.8M、INT8で約50MB/token）は1トークン移動量の約8割を占め、**V維持のまま必ずfactorized head化**
   （1024→256→48,588、INT8で約12.7MB/token、約3.9×減）する。embはINT8 full resident（1行=1KB/tokenで交通費は無視級、容量節約が目的）。
   head疎化の実装はP4に記録のみ。
3. Prefill／Decode／学習の目標は下表§(c)の通り。Decodeは帯域式、Prefillは計算律速（AMX/AVX2）、学習はf2実測のbwd・sync・optimに割付け。

## (a) active上限の再計算

### A1. デコード帯域式

```
tok/s ≒ eff_BW / bytes_per_token
bytes_per_token ≒ active_bytes + kv_bytes_per_token（短文脈では微小）
eff_BW = peak_BW × η（η＝実効効率）
```

- ηの較正：LFM2-24B-A2BはQ4_K_M（スケール込み約0.56B/pと仮定→約1.3GB/token）で112 tok/s。
  112×1.3GB ≒ **145GB/s実効**。Ryzen AI Max+ 395（256-bit LPDDR5x-8000、peak約256GB/s）比で **η≒0.57**。
  Gemma-4前提値（3.8B active・124 tok/s）をINT4（0.5B/p→約1.9GB/token）で換算すると約236GB/s実効。
  いずれも「2〜4B級activeがCPU級帯域で100 tok/s超」という**存在証明**としてのみ扱う（一次ソース未確認の前提値は受領値と明記）。
  語彙が異なる（LFM2は65,536）ためtoks/sの優劣比較には使わない。
- JIMOTONOのη想定：N100（単ch系・小GEMV）で**η=0.5**、Ryzen AI 360で**η=0.5**を採用（LFM2較正0.57より保守的）。
  参考：`analysis/roofline.md` D4は全カーネル帯域律速（算術強度≦0.5、ridge約21）を確定済み。帯域式の適用は正当。

### A2. 上限表（INT4 routed想定、0.5B/p）

| 機体 | peak | eff（η=0.5） | 目標tok/s | byte/token予算 | active上限（INT4） |
|---|---|---|---|---|---|
| N100 | 40GB/s | 20GB/s | 5 | 4.0GB | **約8B** |
| Ryzen AI 360 | 80GB/s（保守前提） | 40GB/s | 30 | 1.33GB | **約2.6B** |

- η=0.1の悲観でもN100上限は約1.6B。**active 100M目標に対する帯域マージンは40〜160×**。
- P3の100M INT4（約50MB）＋縮小head INT8（約12.7MB）＋KV微小 ≒ **約63〜100MB/token**：
  N100で5×100MB=0.5GB/s（effの2.5%）、Ryzenで30×100MB=3GB/s（effの7.5%）。帯域は潤沢。
- したがって律速候補は帯域ではなく：(i)未縮小head（§(b)で除去）、(ii)カーネル起動≦50/tokenの固定費、
  (iii)NVMe読込≦103MiB/token（AGENTS §8）のIOPS・遅延、(iv)スレッドsync（f2実測で6T 6.2%→12T 11.3%）。
- P4への含み：Ryzenの上限約2.6Bは、将来activeを1B級へ拡大する余地が帯域上はあることを示す（実装はP4以降）。

## (b) V維持のままactive拡大 — factorized head＋emb INT8

- 語彙はllm-jp-tokenizer v2.2の**48,588を維持**（`analysis/tokenizer-gap.md`の結論と整合。変更なし）。
- P3のd_model作業仮定=1024（factorized指定1024→256より。proxyのn=256は別物と明記）。

### B1. head縮小の算術（必須）

| 構成 | params | INT8 byte/token | 備考 |
|---|---|---|---|
| Full head 1024×48,588 | 49,754,112 ≒ **49.8M**（≈50M、受領値と一致） | **約49.8MB** | 未縮小だと63〜100MB/token中の約5〜8割＝支配項。FP32なら199MBでN100の5tok/s（4GB予算）すら圧迫 |
| Factorized 1024×256＋256×48,588 | 262,144＋12,438,528＝12,700,672 ≒ **12.7M** | **約12.7MB** | **約3.9×減**。token移動量の約13〜20%に縮小 |
| 削減分 | −37.1M | −37.1MB/token | N100 5tok/sで−186MB/s、Ryzen 30tok/sで−1.1GB/s |

- 設計注意：tied embeddingsとの両立。factorized headはtyingを破るため、P3は**untieを既定**とする：
  embはfull INT8（49.8MB常駐、1行1KB/tokenで交通費無視級）、headのみfactorized。+12.7M paramsの増加は容量予算内（dense≦2GB）。
  対案（factorized-tied）はP3設計会議の記録事項とする。
- head疎化（top-k logit等）の**実装はP4**。本書は記録のみ（タスク禁止事項の遵守）。

### B2. 容量勘定（dense≦2GB INT4／全体≦8GB、AGENTS §8）

- emb INT8 full：約50MB。縮小head INT8：約12.7MB。active 100M INT4：約50MB/token移動、常駐はdense分のみ。
- dense常駐（注意＋共有expert＋emb＋head）のINT4/INT8混成で2GB以内は十分成立（Colibriの17B→9.9GB実績の縮小版としてdispatch可能）。
  詳細な層数・expert数割付けはP3本体設計の入力とする（本書は上限と方針のみ）。

## (c) Prefill／Decode／学習の目標再設定

### C1. Decode目標（帯域式＋Colibri見積り）

| 機体 | 目標 | byte/token想定 | 必要eff_BW | peak比 | 判定 |
|---|---|---|---|---|---|
| N100 | 5 tok/s | 100MB | 0.5GB/s | 1.3% | ◎ 余裕。鍵はI/O重ね合わせ（PIPE/O_DIRECT実測・batch-union・PILOT）と起動回数 |
| Ryzen AI 360 | 30 tok/s | 100MB | 3.0GB/s | 3.8% | ◎ 余裕。W8A8 INT8でさらに半減可 |

- KVはCSA2＋FP4（E2M1、16ch毎E4M3スケール、AGENTS §3.1）で微小化：FP16密の約1/8＋CSA選択（top-1024・window128・heavy128:1）で
  長文脈でもbyte/tokenに効かない設計。KV永続化（`.coli_kv`相当）は追随ターン再prefillゼロ化に寄与。
- NVMe予算103MiB/token（AGENTS §8）対想定100MB/token：**境界上のため**、pin-hit率（`.coli_usage`相当の学習キャッシュ）と
  Delta的持続率（82〜85%前提の実測再取得）で103MiB内に収めることをP3受入条件とする。

### C2. Prefill目標（計算律速）

- 線形注意（KDA／GDN-2、線形:フル=4:1〜7:1）＋conv主体でprefillはGEMM体制（M大）→計算律速。
  T-MAC LUTはroofline D6の判定通り**GEMM体制でのみ有効**（decode GEMVではQLUT構築がamortize不可）につき、prefill・学習を行列側に寄せる。
- SGLang見積りの輸入：W8A8＋AMX＋融合で帯域半減・演算密度向上。P3 prefill目標は「N100で実用TTFT（1k tokens数秒級）」を設計値とし、
  実数はP3実測で確定（本書は式と方針のみ）。

### C3. 学習目標（GEMM＋Colibri見積り、f2実測基準）

基準：`analysis/f2-breakdown.md`（100M-proxy、6T：fwd 6.0%／bwd 46.3%／optim 11.8%／sync 6.2%／ckpt 0.3%／esmoe_io 0.04%、val償却は測定artifactで定常除外）。

| 項目 | P3目標 | 手段（training-speedup.mdの優先度と対応） |
|---|---|---|
| bwd 46%→削減 | 融合SwiGLU（gate/up/down 1カーネル）＋RMSNorm/RoPE融合でメモリパス削減 | P0-3（INT8本線）＋カーネル融合（AGENTS §4.2） |
| optim 12%→削減 | 8bit optimizer（状態1/3.8）＋発火expertのみstep（MEFT疎更新≒SmartUpdate拡張） | P1-1 |
| sync 6→11%の逓増→抑制 | スレッド数適正化＋重ね合わせ（計算／通信／optimizerの優先度スケジューリング、AutoHete型） | P0-2、P1-2 |
| ckpt/esmoe（現0.3%以下） | 10B超の常駐不能域で発動する粒度ポリシー（Megatron式：安価recompute／重量offload＋fraction予約） | P0-4（仕様化のみ） |
| 活性化メモリ | ckptでO(√n)＋FP4/INT8中間＋CPU/SSD退避の併用 | AGENTS §4.1 |
| RL | GRPO/ORPO/SimPO/SPO＋RLVR（AGENTS §4.3維持）。生成N回＋更新1回等の順伝播回数は変更なし | 本書変更なし |

## 禁止事項の遵守記録

- P3実装着手：なし（両文書とも設計入力のみ）。
- 語彙スパース化の実装：なし（P4記録のみ、§(b)）。
- toks/s cross-vocab比較：なし（§(a)方法論宣言の通り帯域正規化のみ。「CPU非現実」での打切りも行わず、上限表で余裕を定量化）。
- コミット：しない（作業ツリー内文書のみ）。

## 参照URL一覧

- `https://huggingface.co/LiquidAI/LFM2-24B-A2B`（active 2.3B・112 tok/s・語彙65,536・Q4_K_M llama.cpp on Ryzen AI Max+ 395）
- `https://www.liquid.ai/blog/lfm2-24b-a2b`
- `https://github.com/JustVugg/colibri`（推定・配置の実測値ソース）
- `analysis/training-speedup.md`（本wt同時作成）
- 既存資産：`analysis/roofline.md`、`analysis/f2-breakdown.md`、`analysis/tokenizer-gap.md`
- Gemma-4 26B MoE（active 3.8B・124 tok/s）はタスク前提値として受領（一次ソース未確認のため本書では帯域換算の存在証明に限定）。
