# llm-from-scratch

A from-scratch, char-level GPT built step by step to learn how LLMs actually
work: tokenization, embeddings, self-attention, the transformer block,
training, and sampling. No pretrained weights, no `transformers` library —
everything is built by hand in PyTorch.

Dataset: [tiny-shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt),
~1.1MB of plain text. Small enough to iterate on fast, large enough to watch
a model actually learn.

## Status

Data pipeline done: download, tokenization, batching. Model code (embeddings,
attention, transformer block, training, sampling) is built incrementally,
one concept per branch. Start at [INSTRUCTIONS.md](INSTRUCTIONS.md).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/data/download.py   # writes data/raw/input.txt
```

## Layout

```
src/
  data/
    download.py   downloads the raw text corpus
    tokenizer.py   text <-> token ids
    batching.py    token ids -> (x, y) training batches
data/
  raw/            downloaded data (gitignored)
```

## Files

### `src/data/download.py`

Fetches the tiny-shakespeare corpus (~1.1MB plain text) and saves it to
`data/raw/input.txt`. This is the raw material everything else operates on —
small enough to iterate on fast, large enough to watch a model actually
learn.

### `src/data/tokenizer.py`

Converts between raw text and lists of integer token ids — a neural network
can't consume strings directly, so this is the first, unavoidable step in
any language model pipeline.

`CharTokenizer` uses the simplest possible scheme: the vocabulary is every
unique character in the corpus (65 for tiny-shakespeare), and each character
maps to one id (`stoi`) and back (`itos`). A string of length N becomes a
list of N ids, one per character.

This is not how production LLMs tokenize — they use subword schemes like BPE
so that common words become a single token instead of many characters, which
keeps sequences short. Char-level is deliberately the easiest possible
starting point: no merge rules to implement, and `decode(encode(text)) ==
text` is trivial to verify as a correctness check.

### `src/data/batching.py`

Turns the flat token stream into `(x, y)` batches for next-token-prediction
training.

- **`train_val_split`**: cuts the token stream once, keeping the last 10%
  held out as validation data. The model never trains on val — it's used
  later (once there's a training loop) to check whether the model is
  generalizing or just memorizing the training text.
- **`get_batch`**: draws `batch_size` random starting offsets into the data
  and slices out a window of `block_size` tokens from each as `x`, plus the
  same window shifted by one position as `y`. Offsets are random and can
  overlap — the stream is *not* chopped into fixed, non-overlapping chunks —
  so every call produces a fresh, decorrelated set of examples to train on.
- Why `y` is `x` shifted by one: at every position `t`, the model sees
  `x[0..t]` and must predict the next token. `y[t]` is that next token,
  i.e. `x[t+1]`. That's the entire supervision signal for training — `x`
  goes into the model, `y` is only used afterwards to score how wrong the
  model's prediction was (cross-entropy loss, added in `08-training-loop`).
