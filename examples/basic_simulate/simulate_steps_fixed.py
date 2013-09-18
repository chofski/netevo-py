#!/usr/bin/env python
"""
NetEvo example showing how a discrete node dynamics can be simulated. A ring
network of identical kuramoto nodes is created and each node given a random
initial state. Nodes are diffusely coupled along the edges and simulation of
the network sees a fast convergence to a synchronised state. Output from the
simulation is printed to the screen.
"""

#=========================================
# IMPORT THE LIBRARIES
#=========================================

# So that we can run the examples without netevo necessarily being 
# in the system path.
import sys
sys.path.append('../netevo')
import netevo
import math
import networkx as nx

#=========================================
# DEFINE THE DYNAMICS
#=========================================

# Define a function for the discrete node dynamics
def kuramoto_node_dyn (G, n, t, y, dy, nmap, emap):
    # Parameters
    natural_freq = 0.1
    coupling_strength = 0.2
    
    # Calculate the new state value
    sum_coupling = 0.0
    for i in G.neighbors(n):
        sum_coupling += math.sin(y[nmap[i]] - y[nmap[n]])
        
    # Calcuate the new state of the node and return the value
    dy[nmap[n]] = math.fmod(y[nmap[n]] + natural_freq + (coupling_strength * 
                                                         sum_coupling), 6.283)

#=========================================
# CREATE THE DYNAMICAL NETWORK
#=========================================

# Create an empty graph (undirected)
G = nx.Graph()

# We only need node dynamics
G.graph['node_dyn'] = True
G.graph['edge_dyn'] = False

# Create the network of n nodes connected in a ring
n_nodes = 4
G.add_node(0)
for i in range(1, n_nodes):
    # Create the node
    G.add_node(i)
    # Connect it to the previous node in the ring
    G.add_edge(i-1, i)
# Finish the ring
G.add_edge(i, 0)

# Set the initial state to a random number in the range (0, 6.0)
netevo.rnd_uniform_node_states (G, [(0.0, 6.0)])
netevo.set_all_node_dynamics(G, kuramoto_node_dyn)

#=========================================
# SIMULATE THE NETWORK DYNAMICS
#=========================================

# Simulate the dynamics and print the state to the screen
res, nmap, emap = netevo.simulate_steps_fixed(G, [0, 50, 100])

# Print the results
print "Simulation Results:"
print '================================='
print "For t = 0:"
print res[0][:]
print '================================='
print "For t = 100:"
print res[2][:]
