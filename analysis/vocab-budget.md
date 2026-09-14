# 語彙パラメータ予算 — untie / factorized の意図明確化

- branch: `docs/p3/pre-g`（本wtのみ。コミットしない）
- 位置づけ: Phase G 事前確認用の文書のみ。実装禁止、重負荷実行なし。
- 目的: `analysis/p3-design-inputs.md` §(b) の untie / factorized 記述に残った曖昧さを除去する。
- スコープ注意: **本書の結論を Phase G に混ぜない。G のスコープは MoE GEMM 化のみ。** 本書は P3 設計入力の明確化である。
- 前提: 語彙 V=48,588（llm-jp-tokenizer v2.2）を維持（`analysis/tokenizer-gap.md`、`p3-design-inputs.md` §(b)と整合。変更なし）。

## 1. 用語定義（曖昧さの除去）

| 用語 | 定義 | 数式（V=48,588、d=d_model） |
|---|---|---|
| emb（入力embedding） | トークンID→ベクトルの写像。全行常駐、1トークンで1行のみ読む | params = V×d |
| head（出力head、full） | 最終隠れ→ロジットの線形写像。デコード毎トークンで全行列を読む | params = d×V |
| tied | emb と head で同一行列を共有（現行 AGENTS.md §2.2 の想定） | 合計 = V×d（共有分のみ） |
| untie-full | emb と head を別行列で持つ。どちらも full | 合計 = 2×V×d |
| factorized head（本書の既定） | head のみ 2 段化：d→k→V。emb は full のまま。tying は構造的に成立しないため untie が必須 | head = d×k＋k×V、語彙合計 = V×d（emb）＋ d×k＋k×V（head） |
| factorized-tied（対案） | factorized した上で emb と何らかの共有を試みる案 | 本書では不採用。P3 設計会議の記録事項とする |

- 既定の組合せは **「emb full ＋ head factorized」の untie** である。「untie かつ full」でも「tied のまま縮小」でもない。
- factorized の中間次元は **k=256** に固定（d=1024 想定時の指定 `1024→256→48,588`）。proxy の n=256 とは別物。
- 精度既定は **INT8**（AGENTS.md §2.2 と整合）。params 数値＝INT8 常駐バイト数に直読できる。

## 2. 試算表（d=64 / 256 / 1024）

V=48,588。INT8 で 1 param = 1 B。

| d | emb V×d | head full d×V | tied 合計 | untie-full 合計 (emb+head) |
|---|---|---|---|---|
| 64 | 48,588×64 = 3,109,632 ≒ **3.11M** | 3,109,632 ≒ **3.11M** | **3.11M**（3.11MB常駐） | **6.22M**（6.22MB常駐） |
| 256 | 48,588×256 = 12,438,528 ≒ **12.44M** | 12,438,528 ≒ **12.44M** | **12.44M** | **24.88M** |
| 1024 | 48,588×1024 = 49,754,112 ≒ **49.75M** | 49,754,112 ≒ **49.75M** | **49.75M** | **99.51M** |

- 常駐と交通費の区別：
  - emb：常駐 V×d（INT8）、**交通費は d B/token**（1行のみ。d=1024 で 1KB/token。無視級）。
  - head full：常駐 d×V、**交通費は d×V B/token**（全行列 GEMV。d=1024 で約 49.8MB/token。支配項）。
  - tied でも交通費の非対称性は変わらない（読む側が emb 行か head 全体かで決まる）。

## 3. factorized head（d=1024、k=256）の削減効果

| 構成（d=1024） | params | INT8 常駐 / 交通費 | 備考 |
|---|---|---|---|
| head full 1024×48,588 | 49,754,112 ≒ **49.75M** | 49.75MB / **約49.8MB/token** | 未縮小。`p3-design-inputs.md` B1 の受領値約50Mと一致 |
| head factorized 1024×256＋256×48,588 | 262,144＋12,438,528 = 12,700,672 ≒ **12.70M** | 12.70MB / **約12.7MB/token**（2段 GEMV の和） | **約3.92×減**（49.75/12.70） |
| 削減分（head単体） | **−37,053,440 ≒ −37.1M** | −37.1MB/token | N100 5tok/s で −186MB/s、Ryzen 30tok/s で −1.1GB/s（`p3-design-inputs.md` B1 と一致） |

語彙全体（emb full を含む）での見え方：

| 語彙全体の組合せ（d=1024） | 合計 params | tied-full（49.75M）比 | untie-full（99.51M）比 |
|---|---|---|---|
| untie ＋ head factorized（既定）：emb 49.75M ＋ head 12.70M | **62,454,784 ≒ 62.45M** | ＋12.70M（共有解除のコストが顕在化） | **−37.05M**（削減効果の正しい比較相手はこちら） |
| tied full（現行想定） | 49.75M | — | — |
| untie full（非推奨の参照点） | 99.51M | — | — |

- 「約12.7M」は **head 単体の値** であり、語彙全体ではない。語彙全体の既定値は **約62.5M** と読む。
- 容量勘定（dense≦2GB／全体≦8GB、AGENTS.md §8）：emb INT8 約50MB ＋ head INT8 約12.7MB の常駐は予算内。＋12.7M の増加分は tied 比で見れば増加だが、untie-full 比では削減であり、予算超過ではない。

## 4. untie を既定にする理由（入出力の役割分離・学習安定）

1. **構造的必然：factorized head は tied と両立しない。** d×V と d×k＋k×V では形状が異なるため、共有するには余分な写像が必要になる。素直な設計は untie である。
2. **役割分離：** emb は入力の分散表現獲得（全トークンで疎に更新）、head は識別的なロジット生成（毎トークン密に評価・密に勾配）。正則化・量子化感度・疎化の方針が異なるため、行列を分け head 側だけ圧縮できる untie が自由度が高い。
3. **学習安定：** tied は emb/head の勾配が同一行列に結合し、head 側の大きな勾配が emb を引っ張る。untie は結合を切り、emb 側を小さい交通費・疎更新のまま安定させる。factorized 側の中間次元 k のスケール設計も head 内で閉じる。
4. **将来拡張：** head 疎化（top-k logit 等）は P4 記録事項（実装禁止の遵守）。emb に触れず head のみ差し替え可能な untie が前提になる。

対案 factorized-tied を既定にしない理由：共有のための追加射影がパラメータ・検証コストを増やし、上記 2〜4 の分離利点を消す。P3 設計会議の記録事項とし、本書では不採用。

## 5. 残存曖昧さの解消チェック

- [x] V=48,588 固定。変更なし。
- [x] d_model 作業仮定=1024。proxy の n=256 と混同しない。
- [x] factorized は head のみ、k=256。emb は full INT8 常駐。
- [x] 既定は untie。AGENTS.md §2.2 の「tied embeddings」表記は本書の既定で上書きされる対象として P3 本体設計へ申送り（本wtでは表記変更しない）。
- [x] head 疎化の実装は P4。記録のみ。
- [x] Phase G への混入なし（G は MoE GEMM 化のみ）。

## 参照

- `analysis/p3-design-inputs.md` §(b)（B1 算術・B2 容量勘定。本書はその意図明確化）
- `analysis/tokenizer-gap.md`（V 維持の根拠）
- AGENTS.md §2.1（V=48,588）、§2.2（混合精度・tied 表記）、§8（dense≦2GB／全体≦8GB）
