#!/usr/bin/env python3
"""llm-jp-tokenizer v2.2 .model -> .jtvocab v1 変換 (Phase F5)。

語彙取得元: https://github.com/llm-jp/llm-jp-tokenizer (release v2.2)
  model: code10K_en20K_ja30K.ver2.2.model (798,534 B, <100MB OK)
  vocab: code10K_en20K_ja30K.ver2.2.vocab (941,665 B, 参考用。直接parseは
         piece内改行のため不可→本scriptは .model protobuf を正本とする)
ライセンス: Apache-2.0 (同repo LICENSE。語彙ファイル自体にLICENSE同梱なし
  のため、本scriptの出力にも版・取得元を埋め込まず、利用時はrepo準拠)。
版: v2.2 / code10K_en20K_ja30K (48,586 pieces。JT_BPE_MAX_ID=48588未満)。

格納対象: NORMAL (type 1) + BYTE (type 6) のみ。
  CONTROL/UNKNOWN/UNUSED は符号化に使わないため除外。
  BYTEピースは flags=1 + 実バイト1Bで格納 (復号時の <0xNN> 逆写像用)。

使い方:
  python3 scripts/bpe_make_vocab.py <in.model> <out.jtvocab>
依存: sentencepiece, protobuf (参照実装の正本読みのみ。C側は依存なし)。
"""
import struct
import sys

MAGIC = b"JTVB"
VERSION = 1
FLAG_BYTE = 1
# ModelProto type enum
T_NORMAL = 1
T_BYTE = 6


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <in.model> <out.jtvocab>",
              file=sys.stderr)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    from sentencepiece import sentencepiece_model_pb2 as m

    mp = m.ModelProto()
    with open(src, "rb") as f:
        mp.ParseFromString(f.read())
    entries = []  # (id, score, flags, bytes)
    max_id = -1
    for i, p in enumerate(mp.pieces):
        if p.type == T_NORMAL:
            b = p.piece.encode("utf-8")
            entries.append((i, float(p.score), 0, b))
            max_id = i
        elif p.type == T_BYTE:
            # piece名 "<0xNN>" から実バイト値を復元
            name = p.piece
            assert name.startswith("<0x") and name.endswith(">"), name
            v = int(name[3:-1], 16)
            entries.append((i, float(p.score), FLAG_BYTE, bytes([v])))
            max_id = i
        else:
            continue  # CONTROL/UNKNOWN/UNUSED を除外
    assert max_id < 48588, max_id
    with open(dst, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, len(entries), max_id))
        f.write(b"\x00" * 16)  # reserved[4] (header計32B)
        for (i, sc, fl, b) in entries:
            f.write(struct.pack("<IfII", i, sc, fl, len(b)))
            f.write(b)
    print(f"bpe_make_vocab: pieces={len(entries)} max_id={max_id} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
