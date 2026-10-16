# pi-gat-methodsx
# PI-GAT: Physics-Informed Graph Attention Network for Power Grid Cyber-Attack Detection

Official implementation of **PI-GAT**, a Physics-Informed Graph Attention Network
for detecting, classifying, and localizing cyber-physical attacks on the IEEE 39-bus
power grid.

The model operates on **6 physics-grounded node features** per bus
(`P_inj, Q_inj, V, θ, r_p, r_q`) — with **no bus-type encoding** — and combines a
supervised classification loss with an AC power-flow physics residual loss.
Attacks are localized at the **node level** across 4 classes:

| Label | Class     | Description                                      |
|-------|-----------|--------------------------------------------------|
| 0     | Normal    | No attack                                        |
| 1     | FDIA      | False Data Injection Attack (stealth, DC-consistent) |
| 2     | Topology  | Line-disconnection / topology attack             |
| 3     | LAA       | Load Altering Attack                             |

---

## Repository Contents

| File | Purpose |
|------|---------|
| `dataset_gen_challenge.py`       | Generates the in-distribution (ID) and out-of-distribution (OOD) datasets. |
| `research_pipeline_challenge.py` | Full experimental pipeline: HP search, 5-fold CV ablation, lambda sweep, ID→OOD generalization, attack-strength sensitivity, robustness, attention visualization, statistical tests. |
| `requirements.txt`               | Python dependencies. |
| `README.md`                      | This file. |

---

## Installation

Tested with **Python 3.9–3.11**.

```bash
# (Recommended) create a fresh environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
