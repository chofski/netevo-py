#!/usr/bin/env python
"""
Shows how the reporter can be used with the matplotlib library to generate 
online visualizations of the current state. The network starts with each 
node having a random state, however, as simulation takes place the system 
becomes locally synchronized.

For large networks we recommend that the visual reporter saves each frame 
to a file which is then combined later to generate a movie.
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

def kuramoto_node_dyn (G, n, t, state):
    # Parameters
    natural_freq = 0.3
    coupling_strength = 0.4
    # Calculate the new state value
    sum_coupling = 0.0
    for i in G.neighbors(n):
        sum_coupling += math.sin(G.node[i]['state'] - state)
    return math.fmod(state + natural_freq + (coupling_strength * 
                                             sum_coupling), 6.283)

#=========================================
# CREATE THE NETWORK
#=========================================
    
# Create an undirected graph with only node dynamics
G = nx.Graph()
G.graph['node_dyn'] = True
G.graph['edge_dyn'] = False

# Graph should have a ring topology
n_nodes = 30
G.add_node(0)
for i in range(1, n_nodes):
    G.add_node(i)
    G.add_edge(i-1, i)
G.add_edge(i, 0)

# All nodes are Kuramoto oscillators (discrete-time)
netevo.set_all_node_dynamics(G, kuramoto_node_dyn)

# Set random initial conditions for each node
netevo.rnd_uniform_node_states (G, [(0.0, 6.2)])

#=========================================
# DEFINE THE VISUAL REPORTER
#=========================================

# Turn on animation in pylab
# http://stackoverflow.com/questions/8965055/basic-animation-with-matplotlibs-pyplot
pylab.ion()
# Create the figure to display the visualization
fig = plt.figure(figsize=(6,6))
# Node positions to use for the visualization
pos=nx.circular_layout(G)

# A visual reporter that will display the current state of the network
# as it is being simulated
def visual_reporter (G, t):
    # Clear the figure
    plt.clf()
    n_sizes = []
    # Calculate the sizes of the nodes (their state)
    for i in G.nodes():
        new_size = 100.0 * G.node[i]['state']
        if new_size < 1.0: new_size = 1
        n_sizes.append(new_size)
    # Draw the network and update the canvas
    nx.draw(G, pos, node_size=n_sizes, node_color='#A0CBE2', width=4, 
            with_labels=False)
    pylab.draw()

#=========================================
# SIMULATE THE DYNAMICS
#=========================================
    
# Simulate the dynamics (discrete-time) using the visual reporter
netevo.simulate_steps(G, 200, visual_reporter)

# Close the visualization
plt.close()
