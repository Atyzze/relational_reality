# Relational Reality: a network evolution engine

**An active research project simulating the emergence of geometry from random networks.**

This engine simulates a "Hamiltonian Routing" dynamic where nodes actively rewire themselves to minimize local stress.

We are mapping the phase transitions that occur when simple local rules give rise to complex global structures.

### ⚡ Quick Start

**1. Install Dependencies**

```bash
pip install -r requirements.txt

```

**2. System Check (`main.py`)**
Run a quick performance test. This does **not** save graph state.

```bash
python main.py

```

* *Default:* `N=100` (Calculates in ~1 min on older hardware).
* *High Res:* `python main.py -N 1000` (Expect longer wait times, but more pixels).

---

### 🧬 Core Workflow

**1. Start the Simulation (`drive.py`)**
This is the main driver. It evolves the topology, finding equilibrium states and storing progress.

```bash
python drive.py

```

**2. Watch it Evolve**
Get immediate feedback on system performance in a separate terminal:

```bash
watch -n 1 python bench1.py

```

**3. Visualize & Analyse**

* **`visualize.py`**: Renders frames from the simulation history to visualize phase transitions.
* **`analyse.py`**: Tallies results to identify trends across parameter regimes.
* **`enhance.py`**: High-resolution zoom for specific windows.

---

### 🔬 Key Findings (So Far)

We have swept the system size up to **N=51200** and identified multiple stable topological phases.

* **Scale Invariance:** The emergent behavior appears identical regardless of size. The only difference is the pixel density; the transitions and geometry remain consistent.
* **Integer Thresholds:** Interesting physics emerge specifically when the average connection count () breaches certain integer values.

### ⚙️ The Physics (Parameters)

You can modify `engine.py` to explore different regimes. The universe is controlled by three primary knobs:

**1. System Size (`N`)**
**2. The Connection Cost (`mu`)**

* A single parameter that controls network density.
* **High** `mu`: High pressure against connections.
* **Low** `mu`: Connections are cheap; the network becomes dense.
* *Observation:* Increasing `mu` suppresses the K-mean (average degree). We are currently mapping exactly what happens as `mu` forces  across integer boundaries.

**3. Non-Local Interaction (`P_TRIADIC_TOGGLE`)**

* How often do we allow non-local connections?
* *Current Setting:* `0.999`. This means we allow "quantum magic" (non-local wiring) only **0.1%** of the time. This tiny fraction is critical, without it, no geometry can ever spawn, it needs connection between at least 2 distinct different points/nodes.

---

### 🌍 Relevance for 2026

Why simulate this? As we move toward massive decentralized systems, mesh networks, and bio-mimetic AI, understanding how **stability emerges from chaos** is critical. This project demonstrates how robust, self-healing architectures can arise naturally from simple energy minimization rules, without a central architect.

---

### 🤝 Call for Collaboration

**This is an invitation.**
There is much more to discover here than one person can compute.

* If you have the compute power to render high-N frames...
* If you can optimize the engine for efficiency...
* If you want to help map the phase diagram...

*Evidently a work in progresSs—~𓆙𓂀*

<img width="3000" height="2550" alt="E14_N51200_S1000_i000_000_000_k0 000" src="https://github.com/user-attachments/assets/e896d5ab-08eb-470e-9b22-e3495a6405d2" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i000_300_000_k0 048" src="https://github.com/user-attachments/assets/fb50cf47-6fa1-43b4-b9f8-879397a19f47" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i002_000_000_k0 322" src="https://github.com/user-attachments/assets/b8e3e75d-1370-4726-b530-596d4fbb6799" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i002_900_000_k0 476" src="https://github.com/user-attachments/assets/f6ddcdaf-97a7-41b4-b4bd-8208f5b96b4e" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i004_500_000_k0 739" src="https://github.com/user-attachments/assets/32bdb0b8-362c-441c-b844-ef7f7330a972" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i004_900_000_k0 801" src="https://github.com/user-attachments/assets/5b911764-f981-4c75-aa7a-abf25b20c4e3" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i006_600_000_k1 058" src="https://github.com/user-attachments/assets/246e82fa-3c59-47a1-b663-24de9e950fa4" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i009_300_000_k1 383" src="https://github.com/user-attachments/assets/366af857-0eb0-422b-a418-8182d3f48277" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i009_500_000_k1 406" src="https://github.com/user-attachments/assets/749b682b-920d-4381-984d-8022b93cb084" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i009_600_000_k1 418" src="https://github.com/user-attachments/assets/a5c33e44-5bed-4865-b5a2-91f46afad9cf" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i012_400_000_k1 665" src="https://github.com/user-attachments/assets/b595ae64-e8ac-4b2e-a43f-f4356aabf76c" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i015_700_000_k1 862" src="https://github.com/user-attachments/assets/d16b1dcc-2a86-42dc-9c9a-e3d2369b2ad3" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i019_300_000_k1 996" src="https://github.com/user-attachments/assets/ac1f71ad-ce8b-4abf-8338-ae57a9e337a6" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i021_100_000_k2 039" src="https://github.com/user-attachments/assets/adcb09a5-7d42-4c6c-902c-589f457010e8" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i023_000_000_k2 076" src="https://github.com/user-attachments/assets/a324daf5-e0c0-4a9a-97eb-0107cf5760a2" />





