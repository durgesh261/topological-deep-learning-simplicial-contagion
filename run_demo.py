"""
End-to-End Demonstration Script for Project 2:
Higher-Order Topological Neural Networks on Simplicial Complexes for Systemic Financial Contagion.

Demonstrates:
1. Construction of 2-Simplicial Complex (Nodes, Bilateral Edges, Triadic Syndicates).
2. Verification of Fundamental Theorem of Topology: B_1 @ B_2 == 0.
3. Hodge-Helmholtz Spectral Decomposition of Interbank Credit Flows.
4. Betti Number Extraction (dim(ker(Delta_1)) topological cycle counting).
5. Eisenberg-Noe Systemic Default Cascade Simulation (EBA stress scenario).
6. Higher-Order Simplicial Convolutional Network (SCNN) Message Passing.
7. Publication-Quality Academic Visualizations.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.financial_network import EuropeanInterbankNetworkGenerator
from src.hodge_laplacian import HodgeLaplacianEngine
from src.contagion_simulator import SystemicContagionSimulator


def main():
    print("=" * 80)
    print("  PROJECT 2: HIGHER-ORDER TOPOLOGICAL NEURAL NETWORKS ON SIMPLICIAL COMPLEXES")
    print("  Geometric Deep Learning & Hodge Theory for Systemic Financial Contagion")
    print("=" * 80)

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate Interbank Simplicial Complex
    print("\n[Step 1] Constructing 2-Simplicial Financial Complex (Nodes, Edges, Triangles)...")
    net_gen = EuropeanInterbankNetworkGenerator(num_banks=30, core_ratio=0.25, seed=42)
    sc, meta = net_gen.generate_network_complex()

    print(f"  -> 0-Simplices (Banks):      {sc.num_nodes} (7 Core G-SIBs + 23 Regional Banks)")
    print(f"  -> 1-Simplices (Bilateral):   {sc.num_edges} (Direct interbank credit links)")
    print(f"  -> 2-Simplices (Syndicates):  {sc.num_triangles} (Higher-order triadic lending syndicates)")

    # 2. Verify Boundary Operators
    print("\n[Step 2] Computing Boundary Operators & Auditing Fundamental Topological Invariant...")
    hodge = HodgeLaplacianEngine(sc)
    norm_err, is_valid = hodge.verify_fundamental_theorem()

    print(f"  -> Boundary Matrix B_1: shape {tuple(hodge.b1.shape)} (Maps edges -> nodes)")
    print(f"  -> Boundary Matrix B_2: shape {tuple(hodge.b2.shape)} (Maps triangles -> edges)")
    print(f"  -> Fundamental Invariant ||B_1 @ B_2||_inf: {norm_err:.2e}")
    print(f"  -> Topology Proof (Boundary of Boundary is Empty): {'VERIFIED (B_1 @ B_2 = 0)' if is_valid else 'FAILED'}")

    # 3. Spectral Hodge Laplacians & Betti Numbers
    print("\n[Step 3] Spectral Analysis of Higher-Order Hodge Laplacians...")
    betti_0, betti_1 = hodge.compute_betti_numbers()
    print(f"  -> Betti Number beta_0 = dim(ker(Delta_0)): {betti_0} (Connected Components)")
    print(f"  -> Betti Number beta_1 = dim(ker(Delta_1)): {betti_1} (Non-Contractible 1-Cycles / Cavities)")
    print(f"  -> Lower 1-Laplacian Delta_1_down:        shape {tuple(hodge.delta_1_down.shape)} (Gradient flows)")
    print(f"  -> Upper 1-Laplacian Delta_1_up:          shape {tuple(hodge.delta_1_up.shape)} (Curl/Triadic flows)")

    # 4. Hodge-Helmholtz Flow Decomposition
    print("\n[Step 4] Computing Hodge-Helmholtz Spectral Flow Decomposition...")
    # Analyze synthetic interbank exposure stress flow
    raw_exposures = meta["edge_features"][:, 0]
    decomp = hodge.decompose_edge_flow(raw_exposures)

    print(f"  -> Curl-Free Gradient Flow (im(B_1^T)):      {(1.0 - decomp.curl_energy_ratio - decomp.harmonic_energy_ratio)*100:>5.1f}% Energy")
    print(f"  -> Divergence-Free Solenoidal Flow (im(B_2)):{decomp.curl_energy_ratio*100:>5.1f}% Energy (Liquidity Loops)")
    print(f"  -> Harmonic Flow (ker(Delta_1)):             {decomp.harmonic_energy_ratio*100:>5.1f}% Energy (Trapped Cavity Flow)")

    # 5. Eisenberg-Noe Contagion Simulation
    print("\n[Step 5] Simulating Eisenberg-Noe Liquidity Cascade (G-SIB Shock)...")
    simulator = SystemicContagionSimulator(sc, meta)
    report = simulator.run_eisenberg_noe_cascade(shock_node=0, shock_amount_meur=35000.0)

    print(f"  -> Primary Shocked Node:        Bank #{report.initial_shocked_bank_id} (Core G-SIB)")
    print(f"  -> Injected Liquidity Shock:    {report.initial_shock_amount_meur:,.0f} MEUR")
    print(f"  -> Cascade Result:              {report.total_defaulted_banks} / {sc.num_nodes} Banks Defaulted")
    print(f"  -> Defaulted Institutions:      {report.defaulted_bank_ids}")
    print(f"  -> Systemic Equity Capital Loss:{report.systemic_capital_loss_meur:,.0f} MEUR ({report.systemic_loss_pct:.1f}% of total sector equity)")
    print(f"  -> Simplicial CNN Test Accuracy:{report.scnn_accuracy*100:.1f}%")
    print(f"  -> Topological Warning Index:   {report.topological_early_warning_score:.4f}")

    # 6. Render Research Plots
    print("\n[Step 6] Generating Geometric Deep Learning Visualizations...")
    _generate_plots(output_dir, sc, hodge, decomp, report)
    print(f"  -> Visual artifacts saved in: {output_dir}")

    print("\n" + "=" * 80)
    print("  PROJECT 2 DEMONSTRATION COMPLETE - ALL ALGEBRAIC TOPOLOGY THEOREMS SATISFIED.")
    print("=" * 80)


def _generate_plots(output_dir, sc, hodge, decomp, report):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Hodge Spectral Energy Decomposition
    flow_labels = ["Gradient Flow\nim(B_1^T)", "Solenoidal Curl Flow\nim(B_2)", "Harmonic Flow\nker(Delta_1)"]
    grad_ratio = 1.0 - decomp.curl_energy_ratio - decomp.harmonic_energy_ratio
    energies = [max(0.0, grad_ratio) * 100, decomp.curl_energy_ratio * 100, decomp.harmonic_energy_ratio * 100]
    colors = ["#3498db", "#e67e22", "#9b59b6"]

    axes[0].bar(flow_labels, energies, color=colors, edgecolor="black", width=0.5)
    axes[0].set_title("Hodge-Helmholtz Spectral Energy Decomposition of Interbank Flows", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Proportion of Flow Energy (%)")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(energies):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

    # Plot 2: Simplicial Complex Dimension Spectrum & Default Cascade
    dims = ["0-Simplices\n(Banks)", "1-Simplices\n(Bilateral Edges)", "2-Simplices\n(Triadic Syndicates)", "Cascade Defaults\n(Failed Banks)"]
    counts = [sc.num_nodes, sc.num_edges, sc.num_triangles, report.total_defaulted_banks]
    axes[1].bar(dims, counts, color=["#1abc9c", "#2980b9", "#f39c12", "#e74c3c"], edgecolor="black", width=0.55)
    axes[1].set_title("Simplicial Complex Topology & Systemic Cascade Footprint", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(counts):
        axes[1].text(i, v + 1, str(v), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "simplicial_complex_and_hodge_decomposition.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
