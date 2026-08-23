"""Turn the token stream into (x, y) training batches.

A language model learns by next-token prediction: given a chunk of tokens,
predict the token that comes right after each position. So for a chunk
`x = tokens[i : i+block_size]`, the target is `y = tokens[i+1 : i+block_size+1]`
— y is just x shifted one position to the right. At every position t,
`y[t]` is the "correct answer" for what should come after `x[0..t]`.

get_batch() builds a random mini-batch of such (x, y) pairs: it picks
`batch_size` random starting offsets into the token stream and slices out
`block_size` tokens from each, plus the same slice shifted by one for the
targets. Random offsets (not sequential chunks) keep training batches
decorrelated across steps.

Train/val split: the last `1 - train_frac` of the corpus is held out and
never trained on, so val loss (checked later, in the training loop) tells
us whether the model generalizes or is just memorizing the training text.
"""

from pathlib import Path

import torch

from src.data.tokenizer import CharTokenizer

DEFAULT_INPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "input.txt"


def train_val_split(data: torch.Tensor, train_frac: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    split_idx = int(len(data) * train_frac)
    return data[:split_idx], data[split_idx:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # a chunk needs block_size+1 tokens available (block_size for x, plus one
    # more for the last target), so offsets can't start past len(data) - block_size - 1
    offsets = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in offsets])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in offsets])
    return x, y


def main() -> None:
    text = DEFAULT_INPUT_PATH.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    train_data, val_data = train_val_split(data)
    print(f"train tokens: {len(train_data):,}")
    print(f"val tokens: {len(val_data):,}")

    block_size, batch_size = 8, 4
    x, y = get_batch(train_data, block_size, batch_size)
    print(f"x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")
    print(f"x[0]: {x[0].tolist()}")
    print(f"y[0]: {y[0].tolist()}")

    # The check from INSTRUCTIONS.md: y is x shifted by one token.
    for i in range(batch_size):
        for t in range(block_size - 1):
            assert x[i][t + 1] == y[i][t]
    print("OK: x[i][t+1] == y[i][t] for every i, t")


if __name__ == "__main__":
    main()
