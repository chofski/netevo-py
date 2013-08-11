#!/usr/bin/env python
"""
In this example we simulate the dynamics of a network with ODE descriptions
for both node and edge states. Nodes are Lorenz chaotic oscillators and edges
use an adaptive law to govern their strength. This is taken from the paper:
 
P. De Lellis et al. "Synchronization of complex networks through local
adaptive coupling". CHAOS 18, 037110 (2008).
 
Starting from a random initial state for both nodes and edges. We simulate
the dynamics to find a final topology achieving synchronisation of the entire
network. The emergence of the topology is animated on screen and the final
topology saved as an image file.
"""

# So that we can run the examples without netevo necessarily being 
# in the system path.
import sys
sys.path.append('../netevo')
import netevo
import math
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab

#=========================================
# DEFINE THE DYNAMICS
#=========================================

# Definition of a chaotic Lorenz oscillator (dimension = 3)
def lorenz_node_dyn (G, n, t, state):   
    # Diffusively couple using the strength of the edge
    c = np.zeros(np.size(state,0))
    for i in G.edges_iter(n):
        c += -G.edge[i[0]][i[1]]['state'] * (G.node[i[1]]['state'] - state)
    # Calculate the derivative
    v1 = (28.0     * (state[1] - state[0]))            - c[0]
    v2 = (state[0] * (10.0 - state[2]) - state[1])     - c[1]
    v3 = (state[0] * state[1] - (8.0/3.0) * state[2])  - c[2]
    return np.array([v1, v2, v3])

# The adaptive edge law defined by De Lellis et al. (dimension = 1)
def adaptive_law_edge_dyn (G, e, t, state):
    s1 = G.node[e[0]]['state']
    s2 = G.node[e[1]]['state']
    dist = np.linalg.norm(s1 - s2)
    return 0.08 * dist

#=========================================
# CREATE THE NETWORK
#=========================================

# Create a fully connected graph
G = nx.complete_graph(6)

# Both node and edge dynamics are required
G.graph['node_dyn'] = True
G.graph['edge_dyn'] = True

# All nodes are chaotic Lorenz oscillators
netevo.set_all_node_dynamics(G, lorenz_node_dyn)
# All edges follow the adaptive rule
netevo.set_all_edge_dynamics(G, adaptive_law_edge_dyn)

# Randomly assign node states
netevo.rnd_uniform_node_states(G, [(0.1, 20.0), (0.1, 20.0), (0.1, 20.0)])
# Edges all start with a very weak strength
netevo.rnd_uniform_edge_states(G, [(0.00000001, 0.00000001)])

for n in range(6):
    G.node[n]['color'] = 'b'

#=========================================
# DEFINE THE VISUAL REPORTER
#=========================================

# Turn on animation in pylab
# http://stackoverflow.com/questions/8965055/basic-animation-with-matplotlibs-pyplot
pylab.ion()
# Create the figure to display the visualization
fig = plt.figure(figsize=(6.5,6.5))


# colors to use for new nodes
colors = 'grcmykw'

# Function to generate the visualisation of the network
def visual_reporter (G, t):
    # Node positions to use for the visualization
    pos = nx.circular_layout(G)
    # Draw the graph
    plt.clf()
    n_sizes = []
    for i in G.nodes():
        new_size = 20.0 * G.node[i]['state'][0] * G.node[i]['state'][1] * G.node[i]['state'][2]
        if new_size < 4.0: new_size = 4
        n_sizes.append(new_size)
    e_sizes = []
    for i in G.edges():
        e_sizes.append(G.edge[i[0]][i[1]]['state'])
    n_colors = []
    for i in G.nodes():
        n_colors.append(G.node[i]['color'])
    nx.draw(G, pos, node_size=n_sizes, node_color=n_colors, 
            edge_color=e_sizes, edge_vmin=0.0, edge_vmax=1.0, width=4,
            edge_cmap=plt.cm.Blues, with_labels=False)
    pylab.draw()
    # Grow the network, by adding nodes and random edges
    if t > 0.3 and t < 2.5:
        if str(t%0.5) == '0.1':
            # Add the node
            n_new = len(G.nodes())
            G.add_node(n_new)
            G.node[n_new]['dyn'] = lorenz_node_dyn
            G.node[n_new]['state'] = [random.uniform(0.1, 20), random.uniform(0.1, 20), random.uniform(0.1, 20)]
            G.node[n_new]['color'] = colors[n_new-6]
            # Add some random edges
            n1 = int(random.uniform(0, n_new))
            n2 = int(random.uniform(0, n_new))
            G.add_edge(n_new, n1)
            G.edge[n_new][n1]['dyn'] = adaptive_law_edge_dyn
            G.edge[n_new][n1]['state'] = 0.00000001
            G.add_edge(n_new, n2)
            G.edge[n_new][n2]['dyn'] = adaptive_law_edge_dyn
            G.edge[n_new][n2]['state'] = 0.00000001

#=========================================
# SIMULATE THE DYNAMICS
#=========================================

# Save initial topology
visual_reporter(G, 0.0) 
plt.savefig('growing_sync_adaptive_edges-start_state.png')

# Simulate the network dynamics
netevo.simulate_rk45(G, 3.0, visual_reporter)

# Save and then close the visualization
plt.savefig('growing_sync_adaptive_edges-end_state.png')
plt.close()
