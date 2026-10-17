"""
Higher-Order Hodge Laplacian & Hodge-Helmholtz Decomposition Module.
Computes Delta_0, Delta_1_down, Delta_1_up, and the 1-Hodge Laplacian Delta_1.
Implements the Hodge-Helmholtz Orthogonal Spectral Decomposition Theorem.
Reference: Schaub et al. (2020) "Random Walks on Simplicial Complexes and the Hodge Laplacian".
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import scipy.linalg as la
import torch
from .simplicial_complex import SimplicialComplex


@dataclass
class HodgeDecompositionResult:
    x_gradient: torch.Tensor     # Curl-free conservative flow in im(B_1^T)
    x_curl: torch.Tensor         # Divergence-free solenoidal flow in im(B_2)
    x_harmonic: torch.Tensor     # Harmonic flow in ker(Delta_1) (Topological cycles)
    betti_0: int                 # Number of connected components
    betti_1: int                 # Number of 1-dimensional topological holes
    curl_energy_ratio: float     # Relative energy in cyclic liquidity loops
    harmonic_energy_ratio: float # Relative energy trapped in structural cavities


class HodgeLaplacianEngine:
    """
    Computes spectral operators on Simplicial Complexes:
    - Delta_0 = B_1 @ B_1^T (Standard 0-Laplacian on nodes)
    - Delta_1_down = B_1^T @ B_1 (Lower 1-Laplacian, gradient dynamics)
    - Delta_1_up = B_2 @ B_2^T (Upper 1-Laplacian, curl/triadic dynamics)
    - Delta_1 = Delta_1_down + Delta_1_up (Full 1-Hodge Laplacian)
    """

    def __init__(self, sc: SimplicialComplex):
        self.sc = sc
        self.b1, self.b2 = sc.compute_boundary_matrices()

        # Compute Laplacians as PyTorch tensors
        self.delta_0 = self.b1 @ self.b1.T
        self.delta_1_down = self.b1.T @ self.b1
        self.delta_1_up = self.b2 @ self.b2.T if sc.num_triangles > 0 else torch.zeros_like(self.delta_1_down)
        self.delta_1 = self.delta_1_down + self.delta_1_up

    def verify_fundamental_theorem(self) -> Tuple[float, bool]:
        """
        Verifies the fundamental topological theorem: B_1 @ B_2 == 0.
        (The boundary of a boundary is identically empty).
        """
        if self.sc.num_triangles == 0:
            return 0.0, True
        b1_np = self.b1.numpy()
        b2_np = self.b2.numpy()
        b1_b2 = b1_np @ b2_np
        norm_err = float(np.max(np.abs(b1_b2)))
        is_valid = norm_err < 1e-6
        return norm_err, is_valid

    def compute_betti_numbers(self, tol: float = 1e-4) -> Tuple[int, int]:
        """
        Computes Betti numbers via nullspace rank of Laplacians:
        - Betti-0 = dim(ker(Delta_0)) = Number of connected components
        - Betti-1 = dim(ker(Delta_1)) = Number of independent 1-cycles (topological holes)
        """
        # Eigenvalues of Delta_0
        eig_0 = np.linalg.eigvalsh(self.delta_0.numpy())
        betti_0 = int(np.sum(eig_0 < tol))

        # Eigenvalues of Delta_1
        eig_1 = np.linalg.eigvalsh(self.delta_1.numpy())
        betti_1 = int(np.sum(eig_1 < tol))

        return betti_0, betti_1

    def decompose_edge_flow(self, edge_flow: torch.Tensor) -> HodgeDecompositionResult:
        """
        Hodge-Helmholtz Orthogonal Decomposition of an edge signal x in R^(N_1):
        x = x_grad + x_curl + x_harm
        where:
        - x_grad in im(B_1^T): Orthogonal projection onto gradient subspace
        - x_curl in im(B_2): Orthogonal projection onto curl subspace
        - x_harm in ker(Delta_1): Harmonic projection (zero curl and zero divergence)
        """
        x_np = edge_flow.detach().cpu().numpy().reshape(-1, 1)
        b1_np = self.b1.numpy()
        b2_np = self.b2.numpy()

        # 1. Gradient projection: x_grad = B_1^T @ (B_1 @ B_1^T)^+ @ B_1 @ x
        pinv_delta0 = la.pinv(self.delta_0.numpy())
        x_grad = b1_np.T @ (pinv_delta0 @ (b1_np @ x_np))

        # 2. Curl projection: x_curl = B_2 @ (B_2^T @ B_2)^+ @ B_2^T @ x
        if self.sc.num_triangles > 0:
            delta2_down = b2_np.T @ b2_np
            pinv_delta2 = la.pinv(delta2_down)
            x_curl = b2_np @ (pinv_delta2 @ (b2_np.T @ x_np))
        else:
            x_curl = np.zeros_like(x_np)

        # 3. Harmonic component: x_harm = x - x_grad - x_curl
        x_harm = x_np - x_grad - x_curl

        betti_0, betti_1 = self.compute_betti_numbers()

        total_energy = float(np.sum(x_np**2)) + 1e-8
        curl_energy = float(np.sum(x_curl**2))
        harm_energy = float(np.sum(x_harm**2))

        return HodgeDecompositionResult(
            x_gradient=torch.tensor(x_grad.squeeze(), dtype=torch.float32),
            x_curl=torch.tensor(x_curl.squeeze(), dtype=torch.float32),
            x_harmonic=torch.tensor(x_harm.squeeze(), dtype=torch.float32),
            betti_0=betti_0,
            betti_1=betti_1,
            curl_energy_ratio=float(curl_energy / total_energy),
            harmonic_energy_ratio=float(harm_energy / total_energy)
        )
