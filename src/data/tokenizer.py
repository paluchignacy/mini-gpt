"""Char-level tokenizer.

The simplest possible tokenizer: the vocabulary is just every unique
character that appears in the corpus. Each character maps to one integer
id and back, so a string of length N becomes a list of N ids.

This is not how production LLMs tokenize (they use subword schemes like
BPE, so common words become a single token instead of many characters —
see the note in INSTRUCTIONS.md). But char-level is the easiest place to
start: no merge rules to implement, and encode/decode are trivial to
verify by round-tripping.
"""

from pathlib import Path

DEFAULT_INPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "input.txt"


class CharTokenizer:
    """Maps between text and lists of integer token ids, one id per character."""

    def __init__(self, text: str) -> None:
        # sorted() makes the vocabulary (and therefore every id) deterministic
        # across runs, given the same corpus.
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # stoi/itos are inverses of each other: stoi maps a character to its
        # id, itos maps that id back to the character.
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def main() -> None:
    text = DEFAULT_INPUT_PATH.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)

    print(f"corpus length: {len(text):,} characters")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"vocab: {''.join(tokenizer.itos.values())!r}")

    sample = text[:50]
    ids = tokenizer.encode(sample)
    print(f"sample text: {sample!r}")
    print(f"encoded ids: {ids}")
    print(f"decoded back: {tokenizer.decode(ids)!r}")

    # The check from INSTRUCTIONS.md: encode/decode must round-trip exactly
    # over the entire corpus, not just the small sample above.
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print("OK: decode(encode(text)) == text for the full corpus")


if __name__ == "__main__":
    main()
