"""Leaky Integrate-and-Fire (LIF) spiking neuron with a surrogate gradient.

This is a minimal, dependency-free reimplementation of the neuron used in
Spikformer (Zhou et al., 2023). Forward pass is a hard Heaviside spike;
backward pass uses a sigmoid-derivative surrogate so the network can be
trained end-to-end with plain autograd / BPTT.
"""

import torch
import torch.nn as nn


SURROGATE_ALPHA = 2.0  # sharpness of the sigmoid surrogate (a plain constant, so torch.compile can trace it)
_ALPHA_TENSOR = None   # set by use_alpha_tensor(): a 0-dim tensor read by the compiled graph, changeable per step


def set_surrogate_alpha(alpha: float):
    """Change the surrogate sharpness (backward only; spikes are unaffected). With a plain float, call it
    before the first compiled forward pass (torch.compile bakes the value in); after use_alpha_tensor()
    it updates the tensor in place, so it can change every step without recompiling."""
    global SURROGATE_ALPHA
    SURROGATE_ALPHA = float(alpha)
    SurrogateSpike.alpha = float(alpha)
    if _ALPHA_TENSOR is not None:
        _ALPHA_TENSOR.fill_(float(alpha))


def use_alpha_tensor(device):
    """Read alpha from a tensor on `device` from now on (for alpha schedules during training)."""
    global _ALPHA_TENSOR
    _ALPHA_TENSOR = torch.tensor(SURROGATE_ALPHA, device=device)


class SurrogateSpike(torch.autograd.Function):
    alpha = SURROGATE_ALPHA

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


def spike_fn(v_minus_threshold: torch.Tensor) -> torch.Tensor:
    """Heaviside spike with the sigmoid-derivative surrogate gradient, written as a straight-through
    expression instead of SurrogateSpike.apply so torch.compile can fuse forward and backward with the
    surrounding LIF arithmetic: the value is exactly the 0/1 step (sg - sg.detach() is exactly 0), the
    gradient is d sigmoid(alpha * x) / dx = alpha * sig * (1 - sig), the same as SurrogateSpike.backward."""
    alpha = _ALPHA_TENSOR if _ALPHA_TENSOR is not None else SURROGATE_ALPHA
    sg = torch.sigmoid(alpha * v_minus_threshold)
    return (v_minus_threshold >= 0).to(v_minus_threshold.dtype) + (sg - sg.detach())


class LIFNeuron(nn.Module):
    """Stateful LIF neuron, called once per simulation time step.

    v[t] = v[t-1] + (x[t] - v[t-1]) / tau
    spike[t] = Heaviside(v[t] - v_threshold)
    v[t] -= spike[t] * v_threshold   (soft reset)

    Call `reset_state()` before starting a new sequence of time steps.

    Multi-step mode (`steps` = T): the input holds all T time steps stacked along
    the batch dimension, (T * B, ...) with time outermost, and the neuron runs the
    T updates itself, starting from v = 0. Everything else in a block (Linear,
    norm, attention) then processes all time steps in one call. The result is
    identical to calling the neuron once per step. With `record_v` set, `v` is left
    holding the post-reset potential of every step, (T * B, ...), for monitoring.
    """

    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v = None
        self.steps = None
        self.record_v = False

    def reset_state(self):
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.steps is not None:
            return self._forward_multistep(x)
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.v + (x - self.v) / self.tau
        spike = spike_fn(self.v - self.v_threshold)
        self.v = self.v - spike * self.v_threshold
        return spike

    def _forward_multistep(self, x: torch.Tensor) -> torch.Tensor:
        xs = x.view(self.steps, -1, *x.shape[1:])
        v = torch.zeros_like(xs[0])
        spikes, vs = [], []
        for x_t in xs:
            v = v + (x_t - v) / self.tau
            spike = spike_fn(v - self.v_threshold)
            v = v - spike * self.v_threshold
            spikes.append(spike)
            if self.record_v:
                vs.append(v.detach())
        self.v = torch.cat(vs) if self.record_v else None
        return torch.cat(spikes)
