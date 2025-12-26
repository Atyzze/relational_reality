sSs... working sstill .. .

# Relational Physics — logs + figures + code

This repo is a self-contained dump of:
- **simulation logs**
- **three diagnostic figures**
- **the Python code** used to generate/run the experiment and plots

It’s meant to be easy to browse first, and runnable second.

---

## Quick start (what to click first)

If you just want the results:

1. **Figures (recommended):**
   - `figures/`  
   Look at these first — they summarize the run.

2. **Logs:**
   - `logs/`  
   Contains raw run outputs / parameters / timing / diagnostics.

3. **Code:**
   - `src/` (or wherever the code lives in this repo)  
   Contains the simulation + analysis scripts used to produce the figures.

---

## What the figures show (high-level)

### 1) Run health + equilibration + correlators
Shows equilibration diagnostics (degree/loops/correlators/energy), correlation vs hop distance, quench response, a “light-cone” style plot, and curvature–density scatter.

### 2) “Einstein test”
Scatter of **Ricci curvature proxy (R)** vs **matter density proxy (T = |psi|^2)** with a linear fit:
- Useful for checking whether curvature responds systematically to matter density.

### 3) Emergent manifold visualization
Two embeddings:
- **Spring layout**: raw connectivity/topology intuition
- **MDS projection**: distance-preserving low-D embedding (what the graph “looks like” as a manifold)

---

## Repo structure

