# llm-from-scratch

A from-scratch, char-level GPT built step by step to learn how LLMs actually
work: tokenization, embeddings, self-attention, the transformer block,
training, and sampling. No pretrained weights, no `transformers` library —
everything is built by hand in PyTorch.

Dataset: [tiny-shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt),
~1.1MB of plain text. Small enough to iterate on fast, large enough to watch
a model actually learn.

## Status

This is the project skeleton: dependencies, folder layout, and the dataset
download script. No model code yet — that's built incrementally, one
concept per branch. Start at [INSTRUCTIONS.md](INSTRUCTIONS.md).

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
data/
  raw/            downloaded data (gitignored)
```
