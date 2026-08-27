"""Self-attention: let each position gather context from earlier ones.

Token+position embeddings (previous branch) give each position a vector,
but those vectors are computed independently — self-attention is what lets
position t pull in information from positions before it, weighted by how
relevant they are.

Each token produces three vectors via learned linear projections:
- query: what this position is looking for
- key: what this position offers to be matched against
- value: what this position actually hands over once matched

Query/key similarity (scaled dot product) sets raw attention scores;
softmax turns them into a distribution that sums to 1 per position; the
output is that distribution's weighted mix of value vectors.

This is a language model — it predicts the next token from the ones before
it — so position t must never attend to positions > t, or it could "cheat"
by looking at the answer it's supposed to predict. A lower-triangular mask
(torch.tril) zeroes out (via -inf before softmax) every score for a future
position.

One head has a single Q/K/V projection, so it can only learn one kind of
relationship between tokens at a time. MultiHeadAttention runs several
smaller heads in parallel (each gets a head_size = n_embd // n_head slice),
so different heads can specialize in different relationships (e.g. "the
previous token" vs. "the matching opening bracket"). Their outputs are
concatenated back to n_embd and mixed by a final linear projection.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Head(nn.Module):
    """One causal self-attention head: (batch, block_size, n_embd) -> (batch, block_size, head_size)."""

    def __init__(self, n_embd: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k = self.key(x)  # (batch, T, head_size)
        q = self.query(x)  # (batch, T, head_size)
        v = self.value(x)  # (batch, T, head_size)

        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size**-0.5  # (batch, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        return wei @ v  # (batch, T, head_size)


class MultiHeadAttention(nn.Module):
    """n_head causal heads run in parallel, then combined back into n_embd.

    Each head only sees a head_size = n_embd // n_head slice of the
    representation, so splitting into more heads doesn't add parameters —
    it trades one head's capacity for several heads that can each learn a
    different kind of token relationship at once.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (batch, T, n_embd)
        return self.proj(out)  # mix information across heads


def main() -> None:
    n_embd, head_size, block_size = 2, 2, 4
    head = Head(n_embd, head_size, block_size)

    # Force Q = K = V = x (identity projections, no bias) so the attention
    # math below can be checked by hand instead of against learned weights.
    with torch.no_grad():
        identity = torch.eye(n_embd)
        head.key.weight.copy_(identity)
        head.query.weight.copy_(identity)
        head.value.weight.copy_(identity)

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]).unsqueeze(
        0
    )  # (1, 4, 2)
    out = head(x)

    print(f"x shape: {tuple(x.shape)}")
    print(f"out shape: {tuple(out.shape)}")
    assert out.shape == (1, block_size, head_size)

    # Hand-computed check (see INSTRUCTIONS.md, branch 04):
    # with Q=K=V=x, scores = x @ x.T / sqrt(head_size), masked causally,
    # softmax per row, then weighted sum of value rows (= x rows here).
    #   t=0: only sees itself                 -> out0 = x0
    #   t=1: sees x0, x1 with scores [0, .707] -> softmax [.330, .670]
    #   t=2: sees x0, x1, x2                   -> softmax [.248, .248, .503]
    #   t=3: all raw scores are 0 (q3 = [0,0]) -> uniform softmax over 4
    expected = torch.tensor(
        [
            [1.0000, 0.0000],
            [0.3302, 0.6698],
            [0.7517, 0.7517],
            [0.5000, 0.5000],
        ]
    ).unsqueeze(0)
    assert torch.allclose(out, expected, atol=1e-4)
    print("OK: output matches the hand-computed attention weights")

    # Branch 05 check: output shape stays (batch, block_size, n_embd)
    # regardless of how many heads n_embd is split across.
    n_embd_mh, block_size_mh = 32, 8
    batch_size = 4
    x_mh = torch.randn(batch_size, block_size_mh, n_embd_mh)
    for n_head in (1, 2, 4, 8):
        mha = MultiHeadAttention(n_embd_mh, n_head, block_size_mh)
        out_mh = mha(x_mh)
        assert out_mh.shape == (batch_size, block_size_mh, n_embd_mh)
    print(
        "OK: MultiHeadAttention output shape is (batch, block_size, n_embd) for n_head in (1, 2, 4, 8)"
    )


if __name__ == "__main__":
    main()
