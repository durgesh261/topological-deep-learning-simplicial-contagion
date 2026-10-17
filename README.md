# Higher-Order Topological Neural Networks on Simplicial Complexes for Systemic Financial Contagion

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Geometric_Deep_Learning_%7C_PyTorch-red.svg)](https://pytorch.org/)
[![Topological Invariant](https://img.shields.io/badge/B_1_@_B_2-0.00e%2B00_%28Exact%29-brightgreen.svg)](#mathematical-formulation)
[![Betti Numbers](https://img.shields.io/badge/Betti_Numbers-β_0%3D1_%7C_β_1%3D6-orange.svg)](#empirical-evaluation)

> **A Geometric Deep Learning framework that extends Graph Neural Networks beyond pairwise edges to higher-order Simplicial Complexes ($\mathcal{K}_0, \mathcal{K}_1, \mathcal{K}_2$). Combines algebraic topology, higher-order Hodge Laplacians ($\Delta_1 = \mathcal{B}_1^\top \mathcal{B}_1 + \mathcal{B}_2 \mathcal{B}_2^\top$), and the Hodge-Helmholtz Orthogonal Decomposition to predict systemic financial default cascades and identify non-contractible liquidity traps.**

---

## 1. Executive Summary & Research Gap

Graph Neural Networks (GNNs: GCN, GAT, GraphSAGE) have emerged as the standard tool for modeling networked systems. However, in complex financial and economic networks (interbank liquidity markets, sovereign debt, CDS counterparty books, and corporate supply chains), standard GNNs suffer from fundamental theoretical limitations:

1. **Restriction to Pairwise Dyadic Interactions:** Standard graphs only represent edges between two vertices ($u \leftrightarrow v$). Financial exposure is inherently **higher-order**: multi-bank syndicates, triadic liquidity swaps, and collateralized debt obligations involve 3 or more institutions acting simultaneously as a **2-simplex (triangle)**.
2. **The 1-Weisfeiler-Lehman (1-WL) Upper Bound:** Pairwise message passing cannot distinguish non-isomorphic graphs that share identical degree distributions and suffers from exponential **over-smoothing and over-squashing** across bottlenecks.
3. **Inability to Detect Topological Cavities:** Standard GNNs cannot detect **topological holes** (non-contractible 1-cycles in harmonic space $\ker(\Delta_1)$) where circular default obligations trap systemic liquidity.

### The Research Contribution
This project addresses this gap by implementing a **Simplicial Complex Neural Network (SCNN)** that models interactions over nodes (0-simplices), bilateral flows (1-simplices), and filled triangles (2-simplices). By computing the higher-order **1-Hodge Laplacian ($\Delta_1$)** and applying the **Hodge-Helmholtz Decomposition Theorem**, the engine decomposes interbank credit stress into conservative gradient flows, circular liquidity loops (curl), and trapped harmonic cavities, detecting systemic contagion prior to cascade onset.

---

## 2. Mathematical Formulation

### 2.1 Oriented Simplicial Complex & Boundary Operators
Let $\mathcal{K} = (\mathcal{K}_0, \mathcal{K}_1, \mathcal{K}_2)$ be an oriented 2-simplicial complex:
* $0$-simplices (Banks): $\mathcal{K}_0 = \{v_i\}_{i=1}^{N_0}$
* $1$-simplices (Bilateral Loans): $\mathcal{K}_1 = \{[v_i, v_j]\}_{j=1}^{N_1}$
* $2$-simplices (Triadic Syndicates): $\mathcal{K}_2 = \{[v_i, v_j, v_k]\}_{k=1}^{N_2}$

We compute the signed boundary matrices:
* $\mathcal{B}_1 \in \{-1, 0, 1\}^{N_0 \times N_1}$: Maps oriented edges to their boundary nodes.
* $\mathcal{B}_2 \in \{-1, 0, 1\}^{N_1 \times N_2}$: Maps oriented triangles to their boundary edges:
  $$\partial_2 [v_i, v_j, v_k] = [v_j, v_k] - [v_i, v_k] + [v_i, v_j]$$

**Fundamental Topological Theorem:** The boundary of a boundary is identically empty:

$$\mathcal{B}_1 \mathcal{B}_2 \equiv 0$$

### 2.2 Higher-Order Hodge Laplacians
While the standard graph Laplacian $\Delta_0 = \mathcal{B}_1 \mathcal{B}_1^\top$ operates purely on nodes, the **1-Hodge Laplacian** governs edge flows:

$$\Delta_1 = \Delta_{1,\text{down}} + \Delta_{1,\text{up}} = \mathcal{B}_1^\top \mathcal{B}_1 + \mathcal{B}_2 \mathcal{B}_2^\top \quad \in \mathbb{R}^{N_1 \times N_1}$$

* $\Delta_{1,\text{down}} = \mathcal{B}_1^\top \mathcal{B}_1$: Lower Laplacian, encoding diffusion through shared vertices.
* $\Delta_{1,\text{up}} = \mathcal{B}_2 \mathcal{B}_2^\top$: Upper Laplacian, encoding circulation through shared triadic 2-simplices.

### 2.3 Hodge-Helmholtz Spectral Decomposition
**Theorem (Hodge):** Any interbank credit flow signal $x \in \mathbb{R}^{N_1}$ admits a unique, mutually orthogonal decomposition:

$$\mathbb{R}^{N_1} = \text{im}(\mathcal{B}_1^\top) \oplus \ker(\Delta_1) \oplus \text{im}(\mathcal{B}_2)$$

$$x = x_{\text{gradient}} + x_{\text{curl}} + x_{\text{harmonic}}$$

1. **Gradient Flow ($x_{\text{grad}} \in \text{im}(\mathcal{B}_1^\top)$):** Curl-free conservative flow driven by potential differences between net-lender and net-borrower banks.
2. **Solenoidal Flow ($x_{\text{curl}} \in \text{im}(\mathcal{B}_2)$):** Divergence-free circular credit loops circulating within closed 2-simplices.
3. **Harmonic Flow ($x_{\text{harm}} \in \ker(\Delta_1)$):** Flow that is simultaneously curl-free and divergence-free. Its dimension equals the **first Betti number**:
   $$\beta_1 = \dim(\ker(\Delta_1))$$
   counting independent topological cavities that trap contagion.

### 2.4 Simplicial Convolutional Network (SCNN) Layer
Higher-order message passing on 1-simplices is parameterized as:

$$H_1^{(l+1)} = \sigma\left( \Delta_{1,\text{down}} H_1^{(l)} W_{\text{grad}} + \Delta_{1,\text{up}} H_1^{(l)} W_{\text{curl}} + H_1^{(l)} W_{\text{self}} \right)$$

Inter-simplex readout to 0-simplices (predicting bank default states) is computed via the absolute boundary projection:

$$H_0 = \text{MLP}\left( |\mathcal{B}_1| H_1^{(L)} \right)$$

---

## 3. German University Curriculum Alignment

This project directly aligns with advanced research areas taught at premier German research universities:

| German University Target Area | Core Scientific Competencies Demonstrated |
| :--- | :--- |
| **TU Munich (TUM) & LMU Munich**<br>*(Munich Center for Machine Learning - MCML)* | Geometric Deep Learning, higher-order message passing, topological graph neural networks. |
| **Heidelberg University**<br>*(Interdisciplinary Center for Scientific Computing - IWR)* | Algebraic topology, Persistent Homology, Hodge Laplacians on cell complexes. |
| **TU Berlin**<br>*(Berlin Institute for the Foundations of Learning and Data - BIFOLD)* | Non-Euclidean data representations, spectral graph theory, systemic network risk. |
| **Paderborn University**<br>*(Heinz Nixdorf Institute & Computer Science)* | Algorithm engineering, simplicial complex indexing, graph algorithms, and PyTest validation. |

---

## 4. Empirical Evaluation & Benchmark Results

Evaluated on a core-periphery European interbank network calibrated to **European Banking Authority (EBA)** and **Bundesbank** stress-testing topology:

| Topological Metric | Measured Value | Theoretical Benchmark | Verification Status |
| :--- | :--- | :--- | :--- |
| **0-Simplices (Banks)** | **30 institutions** | 7 Core G-SIBs + 23 Regional Banks | Complete Core-Periphery |
| **1-Simplices (Bilateral Loans)** | **93 directed edges** | Direct credit exposures | - |
| **2-Simplices (Triadic Syndicates)** | **83 filled triangles** | Co-lending commitments | **Higher-order interactions captured** |
| **Fundamental Boundary Invariant** | **$\|\mathcal{B}_1 \mathcal{B}_2\|_\infty = 0.00e+00$** | Strictly 0 ($\partial \circ \partial = \emptyset$) | **Mathematically Exact** |
| **Betti Number $\beta_0$ (Components)** | **1** | Fully connected network | **Verified** |
| **Betti Number $\beta_1$ (Topological Holes)** | **6 cavities** | Independent non-contractible 1-cycles | **Structural holes detected** |
| **Hodge Gradient Flow Energy** | **42.5%** | Conservative payments ($\text{im}(\mathcal{B}_1^\top)$) | - |
| **Hodge Solenoidal Curl Flow Energy** | **56.8%** | Circulating liquidity loops ($\text{im}(\mathcal{B}_2)$) | **Triadic resonance identified** |
| **Hodge Harmonic Flow Energy** | **0.7%** | Trapped cavity flow ($\ker(\Delta_1)$) | **Topological fragility warning** |
| **Eisenberg-Noe G-SIB Shock Cascade** | **€30,855M Capital Loss** | 9.7% systemic equity wipeout | Cascade simulated |
| **SCNN Default Prediction Accuracy** | **80.0%** | Higher-order simplicial classification | - |

---

## 5. Repository Structure

```
02_Topological_Deep_Learning_Simplicial_Contagion/
├── README.md                          # Academic Research & Mathematical Documentation
├── requirements.txt                   # Dependencies (PyTorch, SciPy, FastAPI)
├── pyproject.toml                     # Modern PEP 517 package config
├── Dockerfile                         # Production containerization
├── docker-compose.yml                 # Local microservice orchestration
├── run_demo.py                        # Complete research trajectory demo
├── app.py                             # Production FastAPI service
├── outputs/
│   └── simplicial_complex_and_hodge_decomposition.png # Spectral decomposition & cascade footprint
├── src/
│   ├── __init__.py
│   ├── simplicial_complex.py          # Abstract Simplicial Complex & Boundary Matrices B_1, B_2
│   ├── hodge_laplacian.py             # Higher-order Hodge Laplacians & Hodge-Helmholtz decomposition
│   ├── topological_nn.py              # Simplicial Convolutional Layer & SCNN model in PyTorch
│   ├── financial_network.py           # European interbank network & syndicated topology generator
│   └── contagion_simulator.py         # Eisenberg-Noe cascade solver & systemic auditor
└── tests/
    ├── test_hodge_laplacian.py        # Pytest verification of B_1@B_2=0 and Hodge orthogonality
    └── test_api.py                    # Pytest verification of FastAPI endpoints
```

---

## 6. How to Run & Deploy

### Quick Start
```bash
cd 02_Topological_Deep_Learning_Simplicial_Contagion
pip install -r requirements.txt
python run_demo.py
python -m pytest tests/ -v
```

### Run FastAPI Microservice
```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```
Interactive Swagger API documentation: [http://localhost:8001/docs](http://localhost:8001/docs)

---

## 7. Author & Contact
* **Applicant:** Durgesh Kanjariya
* **Target:** Master of Science in Data Science / Computer Science in Germany
