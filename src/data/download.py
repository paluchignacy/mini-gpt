"""Download the raw training corpus.

Dataset: tiny-shakespeare (~1.1MB of plain text), the same corpus Karpathy
used for char-rnn / nanoGPT. Small enough to tokenize, batch, and train on
a laptop/CPU while still being large enough to see a language model learn.
"""

import argparse
from pathlib import Path

import requests

DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "input.txt"


def download(url: str = DATASET_URL, output_path: Path = DEFAULT_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    path = download(args.url, args.output)
    text = path.read_text(encoding="utf-8")
    print(f"Saved {len(text):,} characters to {path}")


if __name__ == "__main__":
    main()
