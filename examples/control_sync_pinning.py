#!/usr/bin/env python
"""
In this example we simulate the dynamics of a network with ODE descriptions
for nodes and use pinning based control to allow for a single node to control
the entire network dynamics through only a few connections.
 
Starting from a random initial state for both nodes and no pinning edges, we 
simulate the dynamics to find no correlation between the pinner and the rest
of the network. After a time, pinning edges are added and the network soon
displays the same dynamics as the pinner. The final state is output to file.
"""

# So that we can run the examples without netevo necessarily being 
# in the system path.
import sys
sys.path.append('../netevo')
import netevo
import math
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
    v1 = (28.0    * (state[1] - state[0]))       - c[0]
    v2 = (state[0] * (10.0 - state[2]) - state[1]) - c[1]
    v3 = (state[0] * state[1] - (8.0/3.0) * state[2])  - c[2]
    return np.array([v1, v2, v3])

# Dynamics for the pinner (no coupling)
def pinner_node_dyn (G, n, t, state):
    # Calculate the derivative
    v1 = (28.0    * (state[1] - state[0])) 
    v2 = (state[0] * (10.0 - state[2]) - state[1])
    v3 = (state[0] * state[1] - (8.0/3.0) * state[2])
    return np.array([v1, v2, v3])

#=========================================
# CREATE THE NETWORK
#=========================================

# Create a fully connected graph
#G = nx.complete_graph(12)

G = nx.Graph()
G.add_node(0)
for i in range(1, 12):
    G.add_node(i)
    G.add_edge(i-1, i)
G.add_node(12)

for n in range(12):
    G.node[n]['color'] = 'b'
G.node[12]['color'] = 'r'

# Both node and edge dynamics are required
G.graph['node_dyn'] = True
G.graph['edge_dyn'] = True

# All nodes are chaotic Lorenz oscillators
netevo.set_all_node_dynamics(G, lorenz_node_dyn)
# All edges follow the adaptive rule
netevo.set_all_edge_dynamics(G, netevo.no_edge_dyn)

# Randomly assign node states
netevo.rnd_uniform_node_states (G, [(3.0, 10.0), (3.0, 10.0), (3.0, 10.0)])
# Edges all start with a very weak strength
netevo.rnd_uniform_edge_states (G, [(0.3, 0.3)])

# Our pinning node has no dynamics just fixed states that change
G.node[12]['state'] = [30.0, 30.0, 30.0]
G.node[12]['dyn'] = pinner_node_dyn

#=========================================
# DEFINE THE VISUAL REPORTER
#=========================================

# Turn on animation in pylab
# http://stackoverflow.com/questions/8965055/basic-animation-with-matplotlibs-pyplot
pylab.ion()
# Create the figure to display the visualization
fig = plt.figure(figsize=(6.5,6.5))
# Node positions to use for the visualization
pos=nx.circular_layout(G)

n_colors = []
for i in G.nodes():
    n_colors.append(G.node[i]['color'])

# Function to generate the visualisation of the network
# in addition this also updates the topology during the simulation
def visual_reporter (G, t):
    if t > 0.99 and t < 1.01:
        # Add edge for the pinning (non-adaptive)
        G.add_edge(12, 5)
        G.edge[12][5]['dyn'] = netevo.no_edge_dyn
        G.edge[12][5]['state'] = 0.95
        G.add_edge(12, 2)
        G.edge[12][2]['dyn'] = netevo.no_edge_dyn
        G.edge[12][2]['state'] = 0.95
    if t > 2.99 and t < 3.01:
        G.add_edge(12, 9)
        G.edge[12][9]['dyn'] = netevo.no_edge_dyn
        G.edge[12][9]['state'] = 0.95
        G.node[12]['state'] = np.array([30.0, 30.0, 30.0])
    # Draw the graph
    plt.clf()
    n_sizes = []
    for i in G.nodes():
        new_size = 300.0 * G.node[i]['state'][1]
        if new_size < 1.0: new_size = 1
        n_sizes.append(new_size)
    e_sizes = []
    for i in G.edges():
        if i[1] == 12 and i[0] == 5:
            e_sizes.append(1.0)
        else:
            e_sizes.append(G.edge[i[0]][i[1]]['state'])
    nx.draw(G, pos, node_size=n_sizes, node_color=n_colors, 
            edge_color=e_sizes, edge_vmin=0.0, edge_vmax=1.0, width=4,
            edge_cmap=plt.cm.Blues, with_labels=False)
    pylab.draw()

#=========================================
# SIMULATE THE DYNAMICS
#=========================================
    
# Save initial state
visual_reporter (G, 0.0)
plt.savefig('control_sync_pinning-start_state.png')

# Simulate the network dynamics (this simulator allows for the network size to change)
netevo.simulate_rk45 (G, 7.0, visual_reporter)

# Save and then close the visualization
plt.savefig('control_sync_pinning-end_state.png')
plt.close()
