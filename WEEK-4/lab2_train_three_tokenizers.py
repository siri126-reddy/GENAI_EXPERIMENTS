
"""LAB 2 - Train BPE, WordPiece and SentencePiece
on the same training text.
"""

import os
import tempfile
import sys

import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from corpus import TEST_TEXTS, TRAINING_LINES

sys.stdout.reconfigure(encoding="utf-8")

VOCAB_SIZE = 150


# Train BPE tokenizer
def train_bpe():
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))

    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]"]
    )

    tok.train_from_iterator(TRAINING_LINES, trainer)

    return lambda text: tok.encode(text).tokens


# Train WordPiece tokenizer
def train_wordpiece():
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]"]
    )

    tok.train_from_iterator(TRAINING_LINES, trainer)

    return lambda text: tok.encode(text).tokens


# Train SentencePiece tokenizer
def train_sentencepiece():
    folder = tempfile.mkdtemp()

    data_file = os.path.join(folder, "corpus.txt")

    with open(data_file, "w", encoding="utf-8") as f:
        f.write("\n".join(TRAINING_LINES))

    model_prefix = os.path.join(folder, "sp")

    spm.SentencePieceTrainer.train(
        input=data_file,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        character_coverage=1.0,
        hard_vocab_limit=False,
        minloglevel=2
    )

    sp = spm.SentencePieceProcessor(
        model_file=model_prefix + ".model"
    )

    return lambda text: sp.encode(text, out_type=str)


# Train all three tokenizers
TOKENIZERS = {
    "BPE": train_bpe(),
    "WordPiece": train_wordpiece(),
    "SentencePiece": train_sentencepiece()
}


# Compare tokenizers
summary = []

for language, text in TEST_TEXTS.items():

    print("\n" + "=" * 70)
    print(f"{language}: {text} (characters: {len(text)})")
    print("=" * 70)

    row = [language, len(text)]

    for name, tokenize in TOKENIZERS.items():

        tokens = tokenize(text)

        row.append(len(tokens))

        print(f"{name:14s} ({len(tokens):2d} tokens): {tokens}")

    summary.append(row)


# Display final summary
print("\nSUMMARY - number of tokens (lower = fewer pieces)")

print(
    f"{'Language':10s}"
    f"{'Chars':>7s}"
    f"{'BPE':>8s}"
    f"{'WordPiece':>12s}"
    f"{'SentencePiece':>16s}"
)

for lang, chars, a, b, c in summary:

    print(
        f"{lang:10s}"
        f"{chars:7d}"
        f"{a:8d}"
        f"{b:12d}"
        f"{c:16d}"
    )

print("\nWordPiece marks continuation pieces with ##.")
print("SentencePiece marks word beginnings with ▁.")
