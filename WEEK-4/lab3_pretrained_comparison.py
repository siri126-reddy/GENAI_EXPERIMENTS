
"""LAB 3 - Compare real pretrained tokenizers.
GPT-2 (BPE), mBERT (WordPiece), XLM-R (SentencePiece).
Internet is required for the first run.
"""

import sys
from transformers import AutoTokenizer

from corpus import TEST_TEXTS

sys.stdout.reconfigure(encoding="utf-8")


# Three pretrained models
MODELS = {
    "BPE - GPT-2": "gpt2",
    "WordPiece - mBERT": "bert-base-multilingual-cased",
    "SentencePiece - XLM-R": "xlm-roberta-base"
}


# Load tokenizers
loaded = {}

for label, model_name in MODELS.items():

    print(f"Loading {label} ...")

    try:
        loaded[label] = AutoTokenizer.from_pretrained(model_name)

    except OSError:
        sys.exit(
            f"\nCould not download '{model_name}'. "
            "Check your internet connection and run again."
        )


# Compare tokenizers for each language
summary = []

for language, text in TEST_TEXTS.items():

    print("\n" + "=" * 70)

    print(f"{language}: {text}")

    print(f"Characters: {len(text)}   UTF-8 bytes: {len(text.encode('utf-8'))}")

    print("=" * 70)

    row = [language, len(text)]

    for label, tok in loaded.items():

        # Disable special tokens for fair comparison
        ids = tok(
            text,
            add_special_tokens=False
        )["input_ids"]

        tokens = tok.convert_ids_to_tokens(ids)

        row.append(len(ids))

        print(f"\n{label}")

        print("Tokens :", tokens)

        print("IDs    :", ids)

        print("Count  :", len(ids))

    summary.append(row)


# Display final summary
print("\nSUMMARY - number of tokens")

print(
    f"{'Language':10s}"
    f"{'Chars':>7s}"
    + "".join(f"{label:>24s}" for label in loaded)
)

for row in summary:

    print(
        f"{row[0]:10s}"
        f"{row[1]:7d}"
        + "".join(f"{n:24d}" for n in row[2:])
    )

print("\nGPT-2 uses raw UTF-8 bytes.")
print("Telugu and Hindi may require more tokens with GPT-2.")
