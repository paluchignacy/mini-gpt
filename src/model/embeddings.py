"""Turn token ids into vectors the model can compute with.

Token ids are arbitrary labels (id 47 isn't "more" than id 18) — a network
can't do anything useful with them directly. Embedding fixes this by
looking each id up in a learned table, giving back a dense vector of
n_embd numbers. Those vectors start random and get shaped by training into
something that captures how tokens actually behave in the language.

Self-attention (next branch) has no built-in sense of order — it treats
its input as a set, not a sequence. So position needs its own vector too:
a second lookup table, indexed by position (0..block_size-1) instead of
token identity. Adding the two together gives each position a vector that
encodes both "what token is here" and "where in the window it is".
"""

import torch
from torch import nn

from src.data.batching import DEFAULT_INPUT_PATH, get_batch, train_val_split
from src.data.tokenizer import CharTokenizer


class TokenAndPositionEmbedding(nn.Module):
    """Maps (batch, block_size) token ids to (batch, block_size, n_embd) vectors."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        tok = self.token_emb(idx)  # (batch, block_size, n_embd)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (block_size, n_embd)
        return tok + pos  # (block_size, n_embd) broadcasts over the batch dim


def main() -> None:
    text = DEFAULT_INPUT_PATH.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, _ = train_val_split(data)

    block_size, batch_size, n_embd = 8, 4, 32
    x, _ = get_batch(train_data, block_size, batch_size)

    embed = TokenAndPositionEmbedding(tokenizer.vocab_size, block_size, n_embd)
    out = embed(x)

    print(f"x shape: {tuple(x.shape)}")
    print(f"embedded shape: {tuple(out.shape)}")

    # The check from INSTRUCTIONS.md: (batch, block_size) -> (batch, block_size, n_embd)
    assert out.shape == (batch_size, block_size, n_embd)
    print("OK: output shape is (batch, block_size, n_embd)")


if __name__ == "__main__":
    main()
