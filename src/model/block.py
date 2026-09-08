"""Combine attention and computation into one repeatable transformer block.

MultiHeadAttention (previous branch) lets each position gather context from
earlier ones, but that's purely a *communication* step — a weighted mix of
other positions' value vectors. Nothing in it lets a position think about
what it just gathered. FeedForward adds that: a small per-position MLP that
processes each position's vector independently (no mixing across positions),
giving the model room to compute on the information attention just pulled
in.

A transformer block wires the two together with LayerNorm and residual
("skip") connections around each:

    x = x + MultiHeadAttention(LayerNorm(x))   # communicate
    x = x + FeedForward(LayerNorm(x))          # compute

Two design choices worth naming:
- Pre-norm (LayerNorm before the sub-layer, not after): normalizing the
  input to attention/feedforward keeps their activations well-scaled
  regardless of how deep the stack gets, which is what makes stacking many
  blocks (next branch) trainable at all.
- Residual connections (`x = x + sublayer(x)`, not `x = sublayer(x)`): the
  `+ x` gives gradients a direct path back to earlier layers even if a
  sub-layer's own gradient is small, and lets the network start out close
  to the identity function and gradually learn deviations from it.

Because the block's output has the exact same shape as its input
(batch, block_size, n_embd), blocks can be stacked N deep (branch 07)
without any shape bookkeeping between them.
"""

import torch
from torch import nn

from src.model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Per-position MLP: (batch, block_size, n_embd) -> same shape.

    Expands to 4x n_embd and back down, with a GELU nonlinearity in
    between — the standard GPT-2 sizing. Operates on each position
    independently (no cross-position mixing), giving the model capacity to
    process what attention just gathered.
    """

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block: pre-norm attention + pre-norm feedforward, each with a residual connection."""

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # communicate: gather context via attention
        x = x + self.ff(self.ln2(x))  # compute: process what was gathered
        return x


def main() -> None:
    n_embd, n_head, block_size = 32, 4, 8
    batch_size = 4

    block = Block(n_embd, n_head, block_size)
    x = torch.randn(batch_size, block_size, n_embd)

    # Branch 06 check: the block must be callable repeatedly in a row
    # without changing shape, since branch 07 stacks N of them.
    out = x
    for i in range(3):
        out = block(out)
        assert out.shape == (batch_size, block_size, n_embd), (
            f"shape changed after {i + 1} block(s): {tuple(out.shape)}"
        )

    print(f"x shape: {tuple(x.shape)}")
    print(f"out shape after 3 stacked calls: {tuple(out.shape)}")
    print("OK: shape is unchanged after repeated calls, ready to stack in branch 07")


if __name__ == "__main__":
    main()
