#!/usr/bin/env python3
"""等価性 golden 生成: sentencepiece参照実装の出力を記録 (Phase F5)。

llm-jp-tokenizer は pip 配布なしのため、同一 .model を読む sentencepiece
公式Python実装を参照とする (llm-jp-tokenizer自体がその薄いラッパのため等価)。

使い方:
  python3 scripts/bpe_make_golden.py <in.model> <out.golden>
形式: 1ケース2行 (T:<utf8 hex> / I:id,id,...)。空出力は `I:`。
"""
import sys


CASES = [
    "",
    "Hello world",
    "Hello",
    " Hello",
    "a  b",
    " ",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    "今日はいい天気です。",
    "地元のLLMで推論します。",
    "ツール呼び出し: search(query=\"東京\")",
    "お寿司が食べたい🍣",  # 絵文字=byteフォールバック経路
    "x\ty",
    "a\nb",
    "\n",
    "abc123",
    "v2.2",
    "test日本語",
    "100円",
    "2024年",
    "3.14",
    "C++",
    "foo_bar",
    "end.",
    "a-b",
    "Hello,world",
    "x=42",
    "import os\nprint(os.getpid())  # コメント",
    '{"name": "search", "arguments": {"q": "富士山"}}',
    "<|im_start|>test",  # 特殊記号風文字列は通常テキスト扱い
    "　全角 スペース　混じり",
    "a" * 40,
    "あ" * 40,
    "RustとGoとCの比較: 高速化 15.84x",
    "ﬁ ligature \u00e9",  # 非ASCIIラテン
    "\t leading tab",
    "trailing space ",
    "100x200",
    "a1b",
]


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <in.model> <out.golden>",
              file=sys.stderr)
        return 1
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        for s in CASES:
            ids = sp.encode(s, out_type=int)
            f.write(f"T:{s.encode('utf-8').hex()}\n")
            f.write("I:" + ",".join(str(x) for x in ids) + "\n")
    print(f"bpe_make_golden: cases={len(CASES)} vocab={sp.vocab_size()} "
          f"-> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
