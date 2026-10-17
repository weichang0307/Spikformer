"""Character-level language modelling data (Tiny Shakespeare by default).

The corpus is read as one string, mapped to integer ids through a sorted
character vocabulary, and split by position into train / val (/ test): the
val split follows the train split and the test split is the end of the text.
Tiny Shakespeare uses 90/10 (as in nanoGPT's shakespeare_char); text8 uses
the standard 90M/5M/5M (val_fraction=0.05, test_fraction=0.05).
Batches are random `block_size + 1`-character windows, trained as ordinary
next-token prediction: logits at position t predict window[t + 1].

ASCII corpora (Tiny Shakespeare, text8) are mapped with a byte lookup table
and stored as uint8, so a 100M-character corpus loads in seconds and takes
100 MB of memory; batches are converted to int64 when they are drawn.

If the Tiny Shakespeare file is missing it is downloaded from karpathy/char-rnn.
"""

import os
import urllib.request

import numpy as np
import torch

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_PATH = os.path.join("data", "tinyshakespeare", "input.txt")


class CharDataset:
    def __init__(self, path: str = DEFAULT_PATH, val_fraction: float = 0.1, test_fraction: float = 0.0):
        if not os.path.exists(path):
            if path != DEFAULT_PATH:
                raise FileNotFoundError(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading Tiny Shakespeare to {path}")
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
        with open(path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)

        if raw.size and raw.max() < 128:  # ASCII: byte lookup table, uint8 storage
            codes = np.unique(raw)
            self.chars = [chr(c) for c in codes]
            lut = np.zeros(256, dtype=np.uint8)
            lut[codes] = np.arange(len(codes), dtype=np.uint8)
            data = torch.from_numpy(lut[raw])
        else:
            text = raw.tobytes().decode("utf-8")
            self.chars = sorted(set(text))
            index = {c: i for i, c in enumerate(self.chars)}
            data = torch.tensor([index[c] for c in text], dtype=torch.long)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

        n = len(data)
        n_val, n_test = int(n * val_fraction), int(n * test_fraction)
        n_train = n - n_val - n_test
        self.splits = {"train": data[:n_train], "val": data[n_train:n_train + n_val]}
        if n_test:
            self.splits["test"] = data[n_train + n_val:]

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
        return seq.to(device=device, dtype=torch.long)
