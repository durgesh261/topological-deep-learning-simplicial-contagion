"""
Simplicial Complex Data Structure & Boundary Operators.
Constructs oriented simplicial complexes K = (K_0, K_1, K_2) and calculates
incidence/boundary matrices B_1 and B_2 for algebraic topology and topological deep learning.
Reference: Bodnar et al. (2021) "Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks".
"""

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import scipy.sparse as sp
import torch


class SimplicialComplex:
    """
    Represents an oriented abstract simplicial complex of dimension 2:
    - 0-simplices (Nodes / Entities): V = {v_0, v_1, ...}
    - 1-simplices (Directed Edges / Bilateral Flows): E = {[v_i, v_j]}
    - 2-simplices (Filled Triangles / Syndicates): F = {[v_i, v_j, v_k]}
    """

    def __init__(self):
        self.nodes: List[int] = []
        self.edges: List[Tuple[int, int]] = []
        self.triangles: List[Tuple[int, int, int]] = []

    def build_from_network(
        self,
        num_nodes: int,
        edges: List[Tuple[int, int]],
        fill_triangles: bool = True
    ) -> "SimplicialComplex":
        """
        Builds a 2-simplicial complex (clique complex / flag complex).
        Any 3-cycle of edges automatically constitutes a 2-simplex (triangle).
        """
        self.nodes = list(range(num_nodes))

        # Canonical ordering for orientation: ensure v_i < v_j
        edge_set = set()
        for u, v in edges:
            if u != v:
                edge = (min(u, v), max(u, v))
                edge_set.add(edge)
        self.edges = sorted(list(edge_set))

        # Detect 3-cliques (triangles) to form 2-simplices
        self.triangles = []
        if fill_triangles:
            adj = {i: set() for i in self.nodes}
            for u, v in self.edges:
                adj[u].add(v)
                adj[v].add(u)

            n_edges = len(self.edges)
            for i in range(n_edges):
                u, v = self.edges[i]
                # Common neighbors of u and v
                common = adj[u].intersection(adj[v])
                for w in common:
                    if w > v:  # Canonical ordering u < v < w
                        self.triangles.append((u, v, w))

        self.triangles = sorted(self.triangles)
        return self

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_triangles(self) -> int:
        return len(self.triangles)

    def compute_boundary_matrices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes signed boundary operators:
        1. B_1 in {-1, 0, 1}^(N_0 x N_1):
           B_1[v, e=(u, w)] = +1 if v == w, -1 if v == u, 0 otherwise.
        2. B_2 in {-1, 0, 1}^(N_1 x N_2):
           B_2[e, t=(u, v, w)]: Oriented boundary of 2-simplex.
           B_2 = [v, w] - [u, w] + [u, v].
        
        Fundamental Topological Invariant: B_1 @ B_2 == 0 (Boundary of boundary is zero).
        """
        n0 = self.num_nodes
        n1 = self.num_edges
        n2 = self.num_triangles

        # Map edges to index
        edge_to_idx = {e: idx for idx, e in enumerate(self.edges)}

        # Boundary B_1 (Nodes x Edges)
        b1 = np.zeros((n0, n1), dtype=np.float32)
        for j, (u, v) in enumerate(self.edges):
            b1[u, j] = -1.0  # Tail
            b1[v, j] = 1.0   # Head

        # Boundary B_2 (Edges x Triangles)
        b2 = np.zeros((n1, n2), dtype=np.float32)
        for k, (u, v, w) in enumerate(self.triangles):
            # Edges of oriented triangle (u, v, w): (v, w) - (u, w) + (u, v)
            e_vw = (min(v, w), max(v, w))
            e_uw = (min(u, w), max(u, w))
            e_uv = (min(u, v), max(u, v))

            if e_vw in edge_to_idx:
                b2[edge_to_idx[e_vw], k] = 1.0
            if e_uw in edge_to_idx:
                b2[edge_to_idx[e_uw], k] = -1.0
            if e_uv in edge_to_idx:
                b2[edge_to_idx[e_uv], k] = 1.0

        return torch.tensor(b1, dtype=torch.float32), torch.tensor(b2, dtype=torch.float32)
