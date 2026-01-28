# Relational Reality: a network evolution engine

**An active research project simulating the emergence of geometry from random networks.**

This engine simulates a "Hamiltonian Routing" dynamic where nodes actively rewire themselves to minimize local stress. We are mapping the phase transitions that occur when simple local rules give rise to complex global structures.

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

We have swept the system size from **N=100** up to **N=51,000+** and identified multiple stable topological phases.

* **Scale Invariance:** The emergent behavior appears identical regardless of size. The only difference is the pixel density; the transitions and geometry remain consistent.
* **Integer Thresholds:** Interesting physics emerge specifically when the average connection count () breaches certain integer values.

### ⚙️ The Physics (Parameters)

You can modify `engine.py` to explore different regimes. The universe is controlled by three primary knobs:

**1. System Size (`N`)**

* Range: `100` to `51,200+`.
* *Warning:* Going below `N=100` is untested. You may encounter "Unstable Universes" where the simulation never stabilizes (never triggers the exit condition) and runs forever. If this happens, `Ctrl+C` is your friend.

**2. The Connection Cost (`mu`)**

* A single parameter that controls network density.
* **High `mu**`: High pressure against connections.
* **Low `mu**`: Connections are cheap; the network becomes dense.
* *Observation:* Increasing `mu` suppresses the K-mean (average degree). We are currently mapping exactly what happens as `mu` forces  across integer boundaries.

**3. Non-Local Interaction (`P_TRIADIC_TOGGLE`)**

* How often do we allow non-local connections?
* *Current Setting:* `0.999`. This means we allow "quantum magic" (non-local wiring) only **0.1%** of the time. This tiny fraction is critical—without it, the geometry cannot fold correctly.

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


<img width="3000" height="2550" alt="E14_N51200_S1000_i003_000_000_k0 493" src="https://github.com/user-attachments/assets/d5233129-fc72-4dfa-863a-e91dcbde746f" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i003_900_000_k0 637" src="https://github.com/user-attachments/assets/8dcd757e-ab21-4dc7-986e-c62f3d2aff79" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i004_000_000_k0 654" src="https://github.com/user-attachments/assets/f93cf4d5-9110-41f1-b23d-f5de54d5f525" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i009_100_000_k1 363" src="https://github.com/user-attachments/assets/101a0bd4-6bc9-4c1b-82fa-cf622655b551" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i009_300_000_k1 383" src="https://github.com/user-attachments/assets/2b973013-46d5-4b78-b4eb-b5f329f162e6" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i009_500_000_k1 406" src="https://github.com/user-attachments/assets/26dac3ad-6832-4fa7-8cae-4f477c4071b4" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i012_400_000_k1 665" src="https://github.com/user-attachments/assets/ba5f137e-3eb2-4d67-b910-7edb4bc74582" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i015_700_000_k1 862" src="https://github.com/user-attachments/assets/0684a668-a99b-473b-b334-b4e687da71f2" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i021_100_000_k2 039" src="https://github.com/user-attachments/assets/23d4aec1-e02a-4304-98e9-9dc373837cf2" />

<img width="3000" height="2550" alt="E14_N51200_S1000_i023_000_000_k2 076" src="https://github.com/user-attachments/assets/b906c70e-f952-43a4-a881-443d0d15506a" />

