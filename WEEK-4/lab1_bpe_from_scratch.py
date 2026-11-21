
"""LAB 1 - Byte Pair Encoding (BPE) from scratch.
No internet and no external libraries required.
"""

import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

# Training words
WORDS = [
    "low", "low", "low", "low", "low",
    "lower", "lower",
    "newest", "newest", "newest",
    "newest", "newest", "newest",
    "widest", "widest", "widest"
]

NUM_MERGES = 8

# Convert each word into individual characters
# </w> represents the end of a word
vocab = Counter(tuple(w) + ("</w>",) for w in WORDS)


# Count neighbouring character pairs
def count_pairs(vocab):
    pairs = Counter()

    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq

    return pairs


# Merge the selected pair
def merge_pair(vocab, pair):
    new_vocab = Counter()

    for symbols, freq in vocab.items():
        out, i = [], 0

        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                out.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                out.append(symbols[i])
                i += 1

        new_vocab[tuple(out)] += freq

    return new_vocab


# Tokenize a new word using learned merge rules
def tokenize(word, merges):
    symbols = list(word) + ["</w>"]

    for pair in merges:
        i = 0

        while i < len(symbols) - 1:
            if (symbols[i], symbols[i + 1]) == pair:
                symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]
            else:
                i += 1

    return symbols


# Display training words
print("Training words:", WORDS, "\n")
print("Start:", dict(vocab), "\n")

# Learn merge rules
merges = []

for step in range(1, NUM_MERGES + 1):
    pairs = count_pairs(vocab)

    if not pairs:
        break

    best = max(pairs, key=lambda p: (pairs[p], p))

    merges.append(best)
    vocab = merge_pair(vocab, best)

    print(f"Merge {step}: {best[0]!r} + {best[1]!r} (seen {pairs[best]} times)")
    print("   ", [" ".join(s) for s in vocab])

# Display learned rules
print("\nLearned merge rules (in order):", merges)

# Test new words
print("\nNow tokenize NEW words with those rules:")

for w in ["lowest", "newer", "wider"]:
    print(f" {w:8s} -> {tokenize(w, merges)}")
