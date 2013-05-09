#!/usr/bin/env python
"""
NetEvo example showing how a continuous node dynamics can be simulated. A pair
of identical nodes exhibiting Rossler dynamics is created and each node given 
a random initial state. Nodes are diffusely coupled (on all states) and 
simulation of the network sees a fast convergence to a synchronised state. 
Output from the simulation is printed to the screen. Note that the state for 
the Rossler dynamics is a vector of 3 values which are held in the form of a 
numpy array.
"""

#=========================================
# IMPORT THE LIBRARIES
#=========================================

# So that we can run the examples without netevo necessarily being 
# in the system path.
import sys
sys.path.append('../netevo')
import netevo
import networkx as nx
import numpy as np

#=========================================
# DEFINE THE DYNAMICS
#=========================================

# Define a function for the continuous node dynamics
def lorenz_node_dyn (G, n, t, state):    
    # Parameters
    sigma = 28.0
    rho = 10.0
    beta = 8.0/3.0

    # Calculate the new state value
    c = np.zeros(np.size(state,0))
    coupling = 0.1
    for i in G.neighbors(n):
        c += -coupling * (G.node[i]['state'] - state)
    
    v1 = (sigma    * (state[1] - state[0]))       - c[0]
    v2 = (state[0] * (rho - state[2]) - state[1]) - c[1]
    v3 = (state[0] * state[1] - beta * state[2])  - c[2]
    
    # Return the new state value for the node
    return np.array([v1, v2, v3])

#=========================================
# CREATE THE DYNAMICAL NETWORK
#=========================================

# Create an empty graph (undirected)
G1 = nx.Graph()

# We only need node dynamics
G1.graph['node_dyn'] = True
G1.graph['edge_dyn'] = False

# Create a pair of identical nodes with different initial states
G1.add_node(0)
G1.node[0]['state'] = np.array([2.6, 0.8, 0.9])
G1.node[0]['dyn'] = lorenz_node_dyn
G1.add_node(1)
G1.node[1]['state'] = np.array([1.2, 2.3, 0.2])
G1.node[1]['dyn'] = lorenz_node_dyn

# Connect the nodes with an edge
G1.add_edge(0,1)

#=========================================
# SIMULATE THE NETWORK DYNAMICS
#=========================================

# Simulate the dynamics and print the state to the screen
netevo.simulate_rk45 (G1, 20.0, netevo.state_reporter)
