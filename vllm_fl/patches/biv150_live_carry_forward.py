"""Pinned SDK forward template; checked against the minimal AST rewrite."""
import torch

def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
    hidden_states, residual = _biv150_post_norm_mlp(self, hidden_states, residual)
    return (hidden_states, residual)
