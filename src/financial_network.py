"""
European Interbank Network & Syndicated Lending Topology Generator.
Constructs realistic interbank network matrices calibrated to European Banking Authority (EBA)
and Deutsche Bundesbank systemic stress-testing topology.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import torch
from .simplicial_complex import SimplicialComplex


class EuropeanInterbankNetworkGenerator:
    """
    Synthesizes core-periphery European interbank financial networks:
    - Tier 1 Core: Major European G-SIBs (Deutsche Bank, BNP Paribas, Santander, etc.)
    - Tier 2 Periphery: Regional & specialized banking institutions
    - Triadic syndicates: 2-simplices representing co-lending commitments.
    """

    def __init__(self, num_banks: int = 30, core_ratio: float = 0.25, seed: int = 42):
        self.num_banks = num_banks
        self.num_core = int(num_banks * core_ratio)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_network_complex(self) -> Tuple[SimplicialComplex, Dict[str, Any]]:
        """
        Generates core-periphery topology and populates simplicial financial attributes.
        """
        edges = set()

        # 1. Dense core interconnectivity (probability ~0.80)
        for i in range(self.num_core):
            for j in range(i + 1, self.num_core):
                if self.rng.uniform() < 0.85:
                    edges.add((i, j))

        # 2. Core-to-periphery links (probability ~0.30)
        for i in range(self.num_core):
            for j in range(self.num_core, self.num_banks):
                if self.rng.uniform() < 0.35:
                    edges.add((i, j))

        # 3. Sparse periphery-to-periphery links (probability ~0.08)
        for i in range(self.num_core, self.num_banks):
            for j in range(i + 1, self.num_banks):
                if self.rng.uniform() < 0.08:
                    edges.add((i, j))

        edge_list = sorted(list(edges))

        # Build 2-simplicial complex
        sc = SimplicialComplex()
        sc.build_from_network(num_nodes=self.num_banks, edges=edge_list, fill_triangles=True)

        # Attribute Generation:
        # Bank capital (Tier 1 Equity) in Millions EUR
        bank_capital = np.zeros(self.num_banks)
        bank_capital[: self.num_core] = self.rng.uniform(15000, 45000, size=self.num_core)
        bank_capital[self.num_core :] = self.rng.uniform(2000, 8000, size=self.num_banks - self.num_core)

        # Edge features (1-simplex attributes):
        # [Exposures in MEUR, Tenor in Years, Collateralization Ratio, Credit Spread in bps]
        n_edges = sc.num_edges
        edge_exposures = np.zeros((n_edges, 4), dtype=np.float32)

        for idx, (u, v) in enumerate(sc.edges):
            is_core_link = (u < self.num_core) and (v < self.num_core)
            exp_val = self.rng.uniform(1000, 5000) if is_core_link else self.rng.uniform(100, 1200)
            tenor = self.rng.choice([0.25, 0.5, 1.0, 2.0, 5.0])
            collateral = self.rng.uniform(0.40, 0.95)
            spread_bps = self.rng.uniform(25, 250)

            edge_exposures[idx, :] = [exp_val, tenor, collateral, spread_bps]

        metadata = {
            "num_banks": self.num_banks,
            "num_core_banks": self.num_core,
            "bank_tier1_capital": bank_capital,
            "edge_features": torch.tensor(edge_exposures, dtype=torch.float32)
        }
        return sc, metadata
