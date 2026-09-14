"""Synthetic causal Associative Recall task (the classic induction-head-style
benchmark used e.g. in Ba et al. 2016 and in the H3/Based/Hyena literature).

Each sample presents the same N key-value pairs for R rounds; within each
round the pairs appear in a fresh random order, but a key is always
immediately followed by its paired value:

    round 1: k_a v_a k_b v_b k_c v_c ...   (first sighting, unpredictable)
    round 2: k_c v_c k_a v_a k_b v_b ...   (order reshuffled, but a model that
                                            remembers "k_a -> v_a" can predict
                                            v_a right after seeing k_a again,
                                            regardless of position)
    ...

This is trained as ordinary next-token prediction (causal / autoregressive),
which lets an induction-head mechanism (attend to a previous occurrence of
the current token, copy whatever followed it) solve the task -- this is why
causal self-attention learns associative recall far more easily than a
bidirectional "read the whole sequence, answer at the end" formulation.

Recall accuracy is only meaningful right after a key token, in round >= 2
(the token following a value is a freshly-shuffled key, i.e. unpredictable).
`key_position_mask` marks exactly those positions among the *input*
positions of a next-token-prediction setup (i.e. indices into `seq[:, :-1]`).
"""

import torch


def generate_batch(batch_size: int, num_kv_pairs: int, num_rounds: int,
                    num_keys: int, num_values: int, device="cpu"):
    N = num_kv_pairs
    assert num_keys >= N, "need at least as many possible keys as kv pairs per sample"
    assert num_rounds >= 2, "need at least 2 rounds for recall to be answerable"

    keys = torch.stack([torch.randperm(num_keys)[:N] for _ in range(batch_size)])        # (B, N)
    values = torch.randint(num_keys, num_keys + num_values, (batch_size, N))              # (B, N)

    rounds = []
    for _ in range(num_rounds):
        perm = torch.stack([torch.randperm(N) for _ in range(batch_size)])               # (B, N)
        round_keys = torch.gather(keys, 1, perm)
        round_values = torch.gather(values, 1, perm)
        round_seq = torch.stack([round_keys, round_values], dim=2).reshape(batch_size, 2 * N)
        rounds.append(round_seq)

    seq = torch.cat(rounds, dim=1)  # (B, 2*N*num_rounds)
    return seq.to(device)


def key_position_mask(num_kv_pairs: int, num_rounds: int, device="cpu") -> torch.Tensor:
    L = seq_len(num_kv_pairs, num_rounds)
    pos = torch.arange(L - 1)
    is_key_position = (pos % 2 == 0)
    is_round_2_plus = pos >= 2 * num_kv_pairs
    return (is_key_position & is_round_2_plus).to(device)


def seq_len(num_kv_pairs: int, num_rounds: int) -> int:
    return 2 * num_kv_pairs * num_rounds


def vocab_size(num_keys: int, num_values: int) -> int:
    return num_keys + num_values
