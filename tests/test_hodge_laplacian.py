import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.financial_network import EuropeanInterbankNetworkGenerator
from src.simplicial_complex import SimplicialComplex
from src.hodge_laplacian import HodgeLaplacianEngine


@pytest.fixture
def sample_simplicial_complex():
    net_gen = EuropeanInterbankNetworkGenerator(num_banks=20, core_ratio=0.3, seed=42)
    sc, meta = net_gen.generate_network_complex()
    hodge = HodgeLaplacianEngine(sc)
    return sc, hodge, meta


def test_fundamental_theorem_of_algebraic_topology(sample_simplicial_complex):
    """Proves B_1 @ B_2 == 0 identically (Boundary of a boundary is zero)."""
    sc, hodge, _ = sample_simplicial_complex
    norm_err, is_valid = hodge.verify_fundamental_theorem()

    assert is_valid is True
    assert norm_err < 1e-6


def test_hodge_laplacian_positive_semidefinite(sample_simplicial_complex):
    """Proves that Hodge Laplacians Delta_0 and Delta_1 are positive semi-definite (eigenvalues >= 0)."""
    _, hodge, _ = sample_simplicial_complex

    eig_0 = np.linalg.eigvalsh(hodge.delta_0.numpy())
    eig_1 = np.linalg.eigvalsh(hodge.delta_1.numpy())

    assert np.all(eig_0 >= -1e-5), "Delta_0 has negative eigenvalues"
    assert np.all(eig_1 >= -1e-5), "Delta_1 has negative eigenvalues"


def test_hodge_helmholtz_orthogonality(sample_simplicial_complex):
    """Verifies that gradient and curl components are mutually orthogonal: <x_grad, x_curl> == 0."""
    _, hodge, meta = sample_simplicial_complex
    flow = meta["edge_features"][:, 0]
    decomp = hodge.decompose_edge_flow(flow)

    inner_product = float(torch.dot(decomp.x_gradient, decomp.x_curl))
    assert abs(inner_product) < 1e-3, f"Non-zero inner product between gradient and curl: {inner_product}"
