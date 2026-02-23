import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def run_annealed_replica_mre():
    N = 400
    k0 = 0
    lambda_e = -1.0
    p_triadic = 0.99

    mu_mid = 0.050
    mu_up = np.arange(0.050, 0.1, 0.0005)
    mu_down = np.arange(0.050, 0.025, -0.0005)

    steps_per_mu = 60000

    T_init = 0.01
    cooling_rate = 0.9999

    print(f"1. Warm starting at mu_mid = {mu_mid:.3f} with thermal annealing...")
    G_mid = nx.empty_graph(N)

    def anneal_graph(G, mu):
        T = T_init
        for _ in range(steps_per_mu):
            u = random.choice(list(G.nodes()))
            if random.random() < p_triadic and G.degree(u) > 1:
                neighbors = list(G.neighbors(u))
                v = random.choice(neighbors)
                v1 = random.choice(neighbors)
                while v1 == v and len(neighbors) > 1:
                    v1 = random.choice(neighbors)
                v = v1
            else:
                v = random.choice(list(G.nodes()))
                while v == u:
                    v = random.choice(list(G.nodes()))

            exists = G.has_edge(u, v)
            d_u = G.degree(u)
            d_v = G.degree(v)

            if exists:
                du = 1 - 2 * (d_u - k0)
                dv = 1 - 2 * (d_v - k0)
                dE = -lambda_e + mu * (du + dv)
            else:
                du = 1 + 2 * (d_u - k0)
                dv = 1 + 2 * (d_v - k0)
                dE = lambda_e + mu * (du + dv)

            # THE THERMAL THRESHOLD: Accepts bad moves occasionally when T is high
            if dE <= 0.0 or (T > 1e-10 and random.random() < math.exp(-dE / T)):
                if exists: G.remove_edge(u, v)
                else: G.add_edge(u, v)

            # Cool down
            T *= cooling_rate

        return G

    # Relax the initial state
    G_mid = anneal_graph(G_mid, mu_mid)

    results = {'up': {'mu': [], 'k': [], 'trans': [], 'assort': []},
               'down': {'mu': [], 'k': [], 'trans': [], 'assort': []}}

    print("2. Branching UP (with per-step annealing)...")
    G_up = G_mid.copy()
    for mu in mu_up:
        G_up = anneal_graph(G_up, mu)
        degrees = [d for n, d in G_up.degree()]
        results['up']['mu'].append(mu)
        results['up']['k'].append(np.mean(degrees))
        results['up']['trans'].append(nx.transitivity(G_up))
        try: results['up']['assort'].append(nx.degree_assortativity_coefficient(G_up))
        except: results['up']['assort'].append(0.0)

    print("3. Branching DOWN (with per-step annealing)...")
    G_down = G_mid.copy()
    for mu in mu_down:
        G_down = anneal_graph(G_down, mu)
        degrees = [d for n, d in G_down.degree()]
        results['down']['mu'].append(mu)
        results['down']['k'].append(np.mean(degrees))
        results['down']['trans'].append(nx.transitivity(G_down))
        try: results['down']['assort'].append(nx.degree_assortativity_coefficient(G_down))
        except: results['down']['assort'].append(0.0)

    # --- PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Thermally Annealed Bidirectional Phase Sweep', fontsize=16)

    colors = {'up': 'crimson', 'down': 'royalblue'}

    for dir in ['up', 'down']:
        axs[0, 0].scatter(results[dir]['mu'], results[dir]['k'], s=15, alpha=0.7, color=colors[dir], label=f'Sweep {dir}')
        axs[0, 1].scatter(results[dir]['mu'], results[dir]['trans'], s=15, alpha=0.7, color=colors[dir])
        axs[1, 0].scatter(results[dir]['mu'], results[dir]['assort'], s=15, alpha=0.7, color=colors[dir])

    axs[0, 0].set_title('Mean Degree <k>')
    axs[0, 0].set_ylabel('<k> (Log Scale)')
    axs[0, 0].set_yscale('log')
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].set_title('Transitivity')
    axs[0, 1].grid(alpha=0.3)

    axs[1, 0].set_title('Degree Assortativity')
    axs[1, 0].set_xlabel('mu')
    axs[1, 0].grid(alpha=0.3)

    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_annealed_replica_mre()
