#!/usr/bin/env python
"""
NetEvo example that introduces the concept of evolution using a simulated annealing
metaheuristic. The user must provide functions that quantify a performance
measure (smaller is better - like a cost) and a mutation that should be 
performed to search for new network topologies. The performance function used
here is based on the topological eigenratio measure that is calculated from
the network topology and does not require simulation of the dynamics. Networks
with reduced eigneratios are known to show improved synchronisation.
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
import random
import numpy as np

#=========================================
# DEFINE THE DYNAMICS
#=========================================

# Define a function for the continuous node dynamics
def rossler_node_dyn (G, n, t, y, dy, nmap, emap):    
    # Parameters
    coupling = 0.3
    # Calculate the coupling factor
    c1 = 0.0
    c3 = 0.0
    for i in G.neighbors(n):
        c1 += coupling * (y[nmap[i]] - y[nmap[n]])
        c3 += coupling * (y[nmap[i]+2] - y[nmap[n]+2])
    # Calculate the derivative (include the coupling factors)
    dy[nmap[n]]   = -y[nmap[n]+1] - y[nmap[n]+2] + c1   
    dy[nmap[n]+1] = y[nmap[n]] + 0.165 * y[nmap[n]+1]
    dy[nmap[n]+2] = 0.2 + (y[nmap[n]] - 10.0) * y[nmap[n]+2] + c3 

#=========================================
# CREATE THE NETWORK
#=========================================

# Create a random undirected graph and check connected (n=50, m=100)
n = 50
G = []
while True:
    G = nx.gnm_random_graph(n, 2*n)
    if nx.is_connected(G):
        break

# Only node dynamics are required - should all be chaotic Rossler oscillators
G.graph['node_dyn'] = True
G.graph['edge_dyn'] = False
netevo.set_all_node_dynamics(G, rossler_node_dyn)

#=========================================
# SIMULATE THE NETWORK
#=========================================

# Simulate from random initial conditions
netevo.rnd_uniform_node_states(G, [(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)])

# Simulate the system
res, nmap, emap = netevo.simulate_ode_fixed(G, [0.0, 10.0, 20.0], node_dim=3)

# Print the results
print "Simulation Results:"
print '================================='
print "For t = 0:"
for n in G.nodes():
    print "Node ", n, " = [", res[0][nmap[n]], ', ', res[0][nmap[n]+1], ', ' \
          , res[0][nmap[n]+2], ']' 
print '================================='
print "For t = 20:"
for n in G.nodes():
    print "Node ", n, " = [", res[2][nmap[n]], ', ', res[2][nmap[n]+1], ', ' \
          , res[2][nmap[n]+2], ']' 
