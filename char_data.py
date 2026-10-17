"""Character-level language modelling data (Tiny Shakespeare by default).

The corpus is read as one string, mapped to integer ids through a sorted
character vocabulary, and split 90/10 into train/val by position (the val
split is the last 10% of the text, as in nanoGPT's shakespeare_char).
Batches are random `block_size + 1`-character windows, trained as ordinary
next-token prediction: logits at position t predict window[t + 1].

If the corpus file is missing it is downloaded from karpathy/char-rnn.
"""

import os
import urllib.request

import torch

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_PATH = os.path.join("data", "tinyshakespeare", "input.txt")


class CharDataset:
    def __init__(self, path: str = DEFAULT_PATH, val_fraction: float = 0.1):
        if not os.path.exists(path):
            if path != DEFAULT_PATH:
                raise FileNotFoundError(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading Tiny Shakespeare to {path}")
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
        with open(path, encoding="utf-8") as f:
            text = f.read()

        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n_val = int(len(data) * val_fraction)
        self.splits = {"train": data[:-n_val], "val": data[-n_val:]}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def get_batch(self, split: str, batch_size: int, block_size: int, device="cpu") -> torch.Tensor:
        """(B, block_size + 1) random windows; inputs are [:, :-1], targets [:, 1:]."""
        data = self.splits[split]
        starts = torch.randint(len(data) - block_size - 1, (batch_size,))
        seq = torch.stack([data[s: s + block_size + 1] for s in starts.tolist()])
        return seq.to(device)
