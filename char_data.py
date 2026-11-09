"""Character-level language modelling data (Tiny Shakespeare by default).

The corpus is read as one string, mapped to integer ids through a sorted
character vocabulary, and split by position into train / val (/ test): the
val split follows the train split and the test split is the end of the text.
Tiny Shakespeare uses 90/10 (as in nanoGPT's shakespeare_char); text8 uses
the standard 90M/5M/5M (val_fraction=0.05, test_fraction=0.05). val_size /
test_size give exact split lengths in characters instead (enwik8: the standard
90M/5M/5M byte split decodes to 89,655,885 / 4,982,413 / 4,983,534 characters).
Batches are random `block_size + 1`-character windows, trained as ordinary
next-token prediction: logits at position t predict window[t + 1].

ASCII corpora (Tiny Shakespeare, text8) are mapped with a byte lookup table
and stored as uint8, so a 100M-character corpus loads in seconds and takes
100 MB of memory; batches are converted to int64 when they are drawn.
Other corpora (e.g. Chinese) are mapped through their Unicode code points
with numpy and stored as int16 (or int32). With max_vocab > 0 only the most
frequent max_vocab - 1 characters are kept and every other character becomes
the unknown symbol UNK ("\ufffd"), so rare characters do not blow up the
embedding and output layers.

If the Tiny Shakespeare file is missing it is downloaded from karpathy/char-rnn.
"""

import os
import urllib.request

import numpy as np
import torch

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_PATH = os.path.join("data", "tinyshakespeare", "input.txt")
UNK = "\ufffd"  # stands for characters outside a capped vocabulary


class CharDataset:
    def __init__(self, path: str = DEFAULT_PATH, val_fraction: float = 0.1, test_fraction: float = 0.0,
                 max_vocab: int = 0, val_size: int = 0, test_size: int = 0):
        if not os.path.exists(path):
            if path != DEFAULT_PATH:
                raise FileNotFoundError(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading Tiny Shakespeare to {path}")
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
        with open(path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)

        if raw.size and raw.max() < 128 and not max_vocab:  # ASCII: byte lookup table, uint8 storage
            codes = np.unique(raw)
            self.chars = [chr(c) for c in codes]
            lut = np.zeros(256, dtype=np.uint8)
            lut[codes] = np.arange(len(codes), dtype=np.uint8)
            data = torch.from_numpy(lut[raw])
        else:  # any UTF-8 text: map Unicode code points with numpy
            points = np.frombuffer(raw.tobytes().decode("utf-8").encode("utf-32-le"), dtype=np.uint32)
            codes, inverse, counts = np.unique(points, return_inverse=True, return_counts=True)
            if max_vocab and len(codes) > max_vocab:
                keep = np.sort(np.argsort(-counts, kind="stable")[: max_vocab - 1])  # most frequent, code order
                remap = np.full(len(codes), len(keep), dtype=np.int64)            # everything else -> UNK
                remap[keep] = np.arange(len(keep))
                self.chars = [chr(c) for c in codes[keep]] + [UNK]
                ids = remap[inverse]
                self.unk_fraction = float((ids == len(keep)).mean())
            else:
                self.chars = [chr(c) for c in codes]
                ids = inverse
            dtype = np.uint8 if len(self.chars) <= 256 else np.int16 if len(self.chars) <= 32767 else np.int32
            data = torch.from_numpy(ids.astype(dtype))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

        n = len(data)
        n_val = val_size or int(n * val_fraction)
        n_test = test_size or int(n * test_fraction)
        n_train = n - n_val - n_test
        self.splits = {"train": data[:n_train], "val": data[n_train:n_train + n_val]}
        if n_test:
            self.splits["test"] = data[n_train + n_val:]

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        unk = self.stoi.get(UNK)
        return [self.stoi.get(c, unk) for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    def get_batch(self, split: str, batch_size: int, block_size: int, device="cpu") -> torch.Tensor:
        """(B, block_size + 1) random windows; inputs are [:, :-1], targets [:, 1:]."""
        data = self.splits[split]
        starts = torch.randint(len(data) - block_size - 1, (batch_size,))
        seq = torch.stack([data[s: s + block_size + 1] for s in starts.tolist()])
        return seq.to(device=device, dtype=torch.long)
