"""Leaky Integrate-and-Fire (LIF) spiking neuron with a surrogate gradient.

This is a minimal, dependency-free reimplementation of the neuron used in
Spikformer (Zhou et al., 2023). Forward pass is a hard Heaviside spike;
backward pass uses a sigmoid-derivative surrogate so the network can be
trained end-to-end with plain autograd / BPTT.
"""

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    alpha = 4.0

    @staticmethod
    def forward(ctx, v_minus_threshold):
        ctx.save_for_backward(v_minus_threshold)
        return (v_minus_threshold >= 0).to(v_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = SurrogateSpike.alpha
        sig = torch.sigmoid(alpha * x)
        surrogate_grad = alpha * sig * (1 - sig)
        return grad_output * surrogate_grad


spike_fn = SurrogateSpike.apply


class LIFNeuron(nn.Module):
    """Stateful LIF neuron, called once per simulation time step.

    v[t] = v[t-1] + (x[t] - v[t-1]) / tau
    spike[t] = Heaviside(v[t] - v_threshold)
    v[t] -= spike[t] * v_threshold   (soft reset)

    Call `reset_state()` before starting a new sequence of time steps.
    """

    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v = None

    def reset_state(self):
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.v + (x - self.v) / self.tau
        spike = spike_fn(self.v - self.v_threshold)
        self.v = self.v - spike * self.v_threshold
        return spike
