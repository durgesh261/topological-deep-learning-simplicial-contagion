"""
Simplicial Convolutional Neural Network (SCNN) Module in PyTorch.
Implements higher-order message passing over 0-simplices (nodes) and 1-simplices (edges)
using directional Hodge Laplacian spectral filters (Delta_1_down, Delta_1_up).
Reference: Ebli, Defferrard, Bresson (2020) "Simplicial Neural Networks".
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplicialConvolutionalLayer(nn.Module):
    """
    Higher-order simplicial convolution layer for 1-simplex (edge) feature tensors:
    H_1^(l+1) = sigma( Delta_1_down @ H_1 @ W_grad + Delta_1_up @ H_1 @ W_curl + H_1 @ W_self )
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w_grad = nn.Linear(in_features, out_features, bias=False)
        self.w_curl = nn.Linear(in_features, out_features, bias=False)
        self.w_self = nn.Linear(in_features, out_features, bias=True)

        nn.init.xavier_uniform_(self.w_grad.weight)
        nn.init.xavier_uniform_(self.w_curl.weight)
        nn.init.xavier_uniform_(self.w_self.weight)

    def forward(
        self,
        h1: torch.Tensor,
        delta_down: torch.Tensor,
        delta_up: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs:
        - h1: Tensor of shape (N_edges, in_features)
        - delta_down: Lower 1-Laplacian (N_edges, N_edges)
        - delta_up: Upper 1-Laplacian (N_edges, N_edges)
        """
        # Lower message passing (through shared vertices)
        out_grad = self.w_grad(torch.matmul(delta_down, h1))

        # Upper message passing (through shared triangles / 2-simplices)
        out_curl = self.w_curl(torch.matmul(delta_up, h1))

        # Self transformation
        out_self = self.w_self(h1)

        out = F.silu(out_grad + out_curl + out_self)
        return out


class TopologicalContagionClassifier(nn.Module):
    """
    End-to-end Topological Deep Learning model predicting systemic bank defaults
    by combining Hodge Laplacians with inter-simplex boundary projections.
    """

    def __init__(self, in_edge_dim: int = 4, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        # Simplicial convolutional blocks for 1-simplices
        self.scn1 = SimplicialConvolutionalLayer(in_edge_dim, hidden_dim)
        self.scn2 = SimplicialConvolutionalLayer(hidden_dim, hidden_dim)

        # Boundary projection mapping from 1-simplices (edges) to 0-simplices (nodes)
        self.node_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        edge_features: torch.Tensor,
        delta_down: torch.Tensor,
        delta_up: torch.Tensor,
        b1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        1. Propagate higher-order topological features on edges H_1.
        2. Project edge stress to nodes via absolute incidence matrix: |B_1| @ H_1.
        3. Output node default logits.
        """
        # Normalize Laplacians by spectral radius for stability
        h1 = self.scn1(edge_features, delta_down, delta_up)
        h1 = self.scn2(h1, delta_down, delta_up)

        # Map edge flows to nodes using boundary operator:
        # |B_1| has shape (N_nodes, N_edges)
        abs_b1 = torch.abs(b1)
        node_embeddings = torch.matmul(abs_b1, h1)  # (N_nodes, hidden_dim)

        # Predict node default probabilities
        node_logits = self.node_readout(node_embeddings)
        return node_logits, h1
