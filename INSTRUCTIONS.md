# Plan nauki: budowa LLM od zera

Cel: zbudować mały, char-level model w stylu GPT, krok po kroku, żeby
zrozumieć każdy element (tokenizacja, embeddingi, self-attention,
transformer block, trening, generowanie tekstu). Każdy etap to osobny
branch — commitujesz na nim swój kod, a jak działa, mergujesz do `main` i
przechodzisz dalej.

Szkielet na `main` już istnieje: `requirements.txt`, `.gitignore`,
`src/data/download.py` (ściąga dataset tiny-shakespeare do
`data/raw/input.txt`). Setup:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/data/download.py
```

Punkt odniesienia przez cały czas: [Andrej Karpathy — "Let's build GPT: from
scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
i repo [nanoGPT](https://github.com/karpathy/nanoGPT). Nie kopiuj kodu 1:1 —
oglądaj/czytaj, potem pisz sam, branch po branchu.

---

## Branch `01-tokenization`

**Cel:** zamienić surowy tekst na sekwencję liczb (i z powrotem).

- Zbuduj tokenizer char-level: `chars = sorted(set(text))`, dwa słowniki
  `stoi` / `itos`, funkcje `encode(str) -> list[int]` i
  `decode(list[int]) -> str`.
- Plik: `src/data/tokenizer.py`.
- Sprawdzian: `decode(encode(text)) == text` dla całego korpusu.
- Do przemyślenia (nie trzeba implementować teraz): czym różniłby się
  tokenizer BPE (jak w GPT-2) od char-level — dlaczego produkcyjne LLM-y
  go używają.

## Branch `02-dataset-batching`

**Cel:** zamienić strumień tokenów w batch'e `(x, y)` do treningu.

- Podziel zakodowany tekst na train/val (np. 90/10).
- Napisz `get_batch(split, block_size, batch_size)`: losuje `batch_size`
  fragmentów długości `block_size`, `y` to `x` przesunięte o jeden token
  (next-token prediction).
- Plik: `src/data/batching.py`.
- Sprawdzian: `x[i][t+1] == y[i][t]` dla każdego `i, t`.

## Branch `03-embeddings`

**Cel:** zamienić id tokenu na wektor, dodać informację o pozycji.

- `nn.Embedding(vocab_size, n_embd)` — token embedding.
- `nn.Embedding(block_size, n_embd)` — positional embedding.
- Złóż: `x = token_emb(idx) + pos_emb(arange(T))`.
- Plik: `src/model/embeddings.py`.
- Sprawdzian: dla wejścia `(batch, block_size)` wyjście ma kształt
  `(batch, block_size, n_embd)`.

## Branch `04-self-attention`

**Cel:** jedna głowica self-attention, z maską przyczynowości (causal mask).

- Q, K, V jako liniowe projekcje z `n_embd`.
- `attn = softmax((Q @ K.T) / sqrt(d_k) + causal_mask)`, `out = attn @ V`.
- Maska: token `t` nie może widzieć tokenów `> t` (dolna trójkątna,
  `torch.tril`).
- Plik: `src/model/attention.py`.
- Sprawdzian: ręcznie policz attention na małym przykładzie (3-4 tokeny) i
  porównaj z wynikiem funkcji.

## Branch `05-multi-head-attention`

**Cel:** kilka głowic attention równolegle, złożonych z powrotem w jeden
wektor.

- `n_head` niezależnych głowic z branch `04`, każda mniejsza
  (`n_embd // n_head`), wyniki `concat` + projekcja liniowa.
- Plik: rozszerzenie `src/model/attention.py`.
- Sprawdzian: kształt wyjścia identyczny jak wejścia (`n_embd`), niezależnie
  od `n_head`.

## Branch `06-transformer-block`

**Cel:** złożyć attention + feedforward w jeden powtarzalny blok.

- `LayerNorm -> MultiHeadAttention -> residual`, potem
  `LayerNorm -> FeedForward (MLP z GELU/ReLU) -> residual`.
- Plik: `src/model/block.py`.
- Sprawdzian: blok da się wywołać wielokrotnie z rzędu bez zmiany kształtu
  (żeby dało się je stackować w branchu 07).

## Branch `07-full-model`

**Cel:** złożyć wszystko w model GPT.

- `embeddings -> N x transformer block -> LayerNorm -> linear head (n_embd -> vocab_size)`.
- Klasa `GPTLanguageModel(nn.Module)` w `src/model/gpt.py`, `forward(idx)
  -> logits`.
- Sprawdzian: forward pass na batchu z branch `02` zwraca
  `(batch, block_size, vocab_size)` bez błędów (wagi losowe, strata jeszcze
  wysoka — to normalne).

## Branch `08-training-loop`

**Cel:** faktyczny trening.

- Loss: `cross_entropy(logits.view(-1, vocab_size), targets.view(-1))`.
- Optimizer: `AdamW`. Pętla treningowa z okresową ewaluacją na val split.
- Plik: `src/train.py` (entrypoint, argparse na hyperparametry).
- Sprawdzian: strata treningowa systematycznie spada z epoki na epokę /
  z iteracji na iterację.

## Branch `09-sampling-generation`

**Cel:** generować tekst z wytrenowanego modelu.

- `generate(model, idx, max_new_tokens)`: w pętli — forward, weź logity
  ostatniej pozycji, `softmax`, `torch.multinomial` (opcjonalnie
  temperature / top-k), dołóż token, powtórz.
- Plik: `src/generate.py`.
- Sprawdzian: model po treningu generuje tekst, który *wygląda* jak
  Szekspir (składnia, formatowanie dialogów) nawet jeśli nie ma sensu
  semantycznego — to oczekiwany wynik małego char-level modelu.

## Branch `10-evaluation`

**Cel:** zmierzyć jakość liczbowo, nie tylko "na oko".

- Perplexity na val split (`exp(val_loss)`).
- Krzywe train/val loss w czasie (matplotlib albo tensorboard — do wyboru).
- Plik: `src/evaluate.py`.
- Sprawdzian: masz wykres/log pokazujący, że val loss przestaje spadać w
  pewnym momencie (overfitting) — to punkt, w którym warto się zatrzymać
  albo zwiększyć dataset/model.

---

## Kolejność pracy na branchu

1. `git checkout main && git checkout -b <nazwa-brancha>`
2. Implementuj + testuj lokalnie (mały skrypt/`python -i` wystarczy, nie
   trzeba pytest na tym etapie).
3. Jak sprawdzian z sekcji wyżej przechodzi: `git add`, commit.
4. `git checkout main && git merge <nazwa-brancha>`.
5. Następny branch.

Branch `07` i dalej zależą od kodu z branchy `03`–`06` — mergować po kolei,
nie przeskakiwać.
