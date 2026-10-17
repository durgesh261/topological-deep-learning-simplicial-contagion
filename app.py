"""
Production FastAPI Application for Higher-Order Topological Neural Networks.
Geometric Deep Learning & Hodge Theory for Systemic Financial Contagion.

Endpoints:
- GET  /                           : System health & simplicial complex statistics
- GET  /api/v1/topology            : Betti numbers (beta_0, beta_1) and boundary operator ranks
- GET  /api/v1/hodge-decomposition : Hodge-Helmholtz gradient vs curl vs harmonic flow decomposition
- POST /api/v1/simulate-cascade    : Injects liquidity shock & runs Eisenberg-Noe cascade + SCNN
"""

import os
import sys
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.financial_network import EuropeanInterbankNetworkGenerator
from src.simplicial_complex import SimplicialComplex
from src.hodge_laplacian import HodgeLaplacianEngine
from src.contagion_simulator import SystemicContagionSimulator


# --- Global Model Context ---
context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes European Interbank Simplicial Complex and Hodge Engine on startup."""
    print("[API Startup] Initializing 2-Simplicial Complex and Hodge Laplacians...")
    net_gen = EuropeanInterbankNetworkGenerator(num_banks=30, core_ratio=0.25, seed=42)
    sc, meta = net_gen.generate_network_complex()
    hodge = HodgeLaplacianEngine(sc)
    simulator = SystemicContagionSimulator(sc, meta)

    context["net_gen"] = net_gen
    context["sc"] = sc
    context["meta"] = meta
    context["hodge"] = hodge
    context["simulator"] = simulator

    print("[API Startup] Topological Deep Learning microservice ready.")
    yield
    context.clear()


app = FastAPI(
    title="Topological Deep Learning & Simplicial Contagion API",
    description=(
        "Geometric Deep Learning microservice based on Higher-Order Hodge Theory and Simplicial Complexes. "
        "Models systemic financial cascades across nodes (0-simplices), bilateral flows (1-simplices), "
        "and triadic syndicates (2-simplices) with Hodge-Helmholtz orthogonal spectral decomposition."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---

class ShockRequest(BaseModel):
    shock_bank_id: int = Field(0, json_schema_extra={"example": 0}, description="Target bank index (0 to 6 are Core G-SIBs)")
    shock_amount_meur: float = Field(35000.0, json_schema_extra={"example": 35000.0}, description="Injected liquidity loss in Millions EUR")


# --- Endpoints ---

@app.get("/", tags=["System"])
def root():
    sc: SimplicialComplex = context["sc"]
    hodge: HodgeLaplacianEngine = context["hodge"]
    betti_0, betti_1 = hodge.compute_betti_numbers()

    return {
        "service": "Topological Deep Learning & Simplicial Contagion Engine",
        "geometric_structure": "2-Simplicial Complex (Flag/Clique Complex)",
        "simplices": {
            "0_simplices_nodes": sc.num_nodes,
            "1_simplices_edges": sc.num_edges,
            "2_simplices_triangles": sc.num_triangles
        },
        "betti_numbers": {
            "betti_0_connected_components": betti_0,
            "betti_1_topological_holes": betti_1
        },
        "documentation": "/docs",
        "author": "Durgesh Kanjariya"
    }


@app.get("/api/v1/topology", tags=["Algebraic Topology"])
def get_topology_metrics():
    """Returns boundary matrix ranks, Hodge Laplacian dimensions, and Betti numbers."""
    sc: SimplicialComplex = context["sc"]
    hodge: HodgeLaplacianEngine = context["hodge"]
    betti_0, betti_1 = hodge.compute_betti_numbers()
    norm_err, is_valid = hodge.verify_fundamental_theorem()

    return {
        "num_nodes": sc.num_nodes,
        "num_edges": sc.num_edges,
        "num_triangles": sc.num_triangles,
        "betti_0": betti_0,
        "betti_1": betti_1,
        "boundary_invariant_norm": norm_err,
        "fundamental_theorem_holds": is_valid,
        "delta_0_shape": list(hodge.delta_0.shape),
        "delta_1_shape": list(hodge.delta_1.shape)
    }


@app.get("/api/v1/hodge-decomposition", tags=["Hodge-Helmholtz Theory"])
def get_hodge_decomposition():
    """Decomposes interbank credit flows into gradient (curl-free), solenoidal (divergence-free), and harmonic subspaces."""
    hodge: HodgeLaplacianEngine = context["hodge"]
    meta = context["meta"]
    raw_flow = meta["edge_features"][:, 0]
    decomp = hodge.decompose_edge_flow(raw_flow)

    grad_pct = max(0.0, 1.0 - decomp.curl_energy_ratio - decomp.harmonic_energy_ratio) * 100.0
    curl_pct = decomp.curl_energy_ratio * 100.0
    harm_pct = decomp.harmonic_energy_ratio * 100.0

    return {
        "gradient_flow_pct": round(grad_pct, 2),
        "solenoidal_curl_flow_pct": round(curl_pct, 2),
        "harmonic_cavity_flow_pct": round(harm_pct, 2),
        "betti_1_independent_holes": decomp.betti_1,
        "interpretation": {
            "gradient": "Conservative payment flow between surplus and deficit banks",
            "curl": "Circular credit/liquidity loops trapped in 2-simplex syndicates",
            "harmonic": "Non-contractible topological cycles indicating systemic fragility"
        }
    }


@app.post("/api/v1/simulate-cascade", tags=["Contagion Simulation"])
def simulate_cascade(req: ShockRequest):
    """Simulates Eisenberg-Noe default clearing cascade and evaluates SCNN predictions."""
    sc: SimplicialComplex = context["sc"]
    if not (0 <= req.shock_bank_id < sc.num_nodes):
        raise HTTPException(status_code=400, detail=f"Invalid bank_id. Must be in [0, {sc.num_nodes - 1}]")

    simulator: SystemicContagionSimulator = context["simulator"]
    report = simulator.run_eisenberg_noe_cascade(
        shock_node=req.shock_bank_id,
        shock_amount_meur=req.shock_amount_meur
    )

    return {
        "shocked_bank": req.shock_bank_id,
        "shock_amount_meur": req.shock_amount_meur,
        "defaulted_banks_count": report.total_defaulted_banks,
        "defaulted_bank_ids": [int(x) for x in report.defaulted_bank_ids],
        "systemic_equity_loss_meur": round(report.systemic_capital_loss_meur, 2),
        "systemic_equity_loss_pct": round(report.systemic_loss_pct, 2),
        "topological_early_warning_score": round(report.topological_early_warning_score, 4),
        "scnn_prediction_accuracy_pct": round(report.scnn_accuracy * 100.0, 1)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
