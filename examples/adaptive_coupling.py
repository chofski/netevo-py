#!/usr/bin/python
"""
adaptive_coupling.py

In this example we simulate the dynamics of a network with ODE descriptions
for both node and edge states. Nodes are Lorenz chaotic oscillators and edges
use an adaptive law to govern their strength. This is taken from the paper:
 
P. De Lellis et al. "Synchronization of complex networks through local
adaptive coupling". CHAOS 18, 037110 (2008).
 
Starting from a random initial state for both nodes and edges. We simulate
the dynamics to find a final topology achieving synchronisation of the entire
network. Finally, we copy the resting edge states to the weight variable
of edges in our system and export to file. This can then be viewed within
tools such as Cytoscape - http://www.cytoscape.org
"""

# So that we can run the examples without netevo necessarily being 
# in the system path.
import sys
sys.path.append('../netevo-py')
import netevo

import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt





def lorenz_node_dyn (G, n, t, state):	
	# Parameters
	cur_params = G.node[n]['params']
	sigma = cur_params[0]
	rho = cur_params[1]
	beta = cur_params[2]

	# Diffusively couple using the strength of the edge
	c = np.zeros(np.size(state,0))
	for i in G.edges_iter(n):
		c += -G.edge[i[0]][i[1]]['state'] * (G.node[i[1]]['state'] - state)
	
	v1 = (sigma    * (state[1] - state[0]))       - c[0]
	v2 = (state[0] * (rho - state[2]) - state[1]) - c[1]
	v3 = (state[0] * state[1] - beta * state[2])  - c[2]

	return np.array([v1, v2, v3])


def adaptive_law_edge_dyn (G, source, target, t, state):

	s1 = G.node[source]['state']
	s2 = G.node[target]['state']
	
	d =  math.pow(s2[0] - s1[0], 2.0)
	d += math.pow(s2[1] - s1[1], 2.0)
	d += math.pow(s2[2] - s1[2], 2.0)
	
	return 0.1 * math.sqrt(d)




# Test the discrete dynamics

fig = plt.figure()

def visual_reporter (G, t):
	"""
	Standard simulation state reporter that outputs the current time and
	node states for the system.
	"""
	print t
	plt.clf()
	pos=nx.circular_layout(G)
	n_sizes = []
	for i in G.nodes():
		n_sizes.append(150.0 * G.node[i]['state'][2])
	nx.draw(G, pos, node_size=n_sizes)
	nx.draw_networkx_edge_labels(G,pos, edge_labels='weight')
	fig.canvas.draw()



G2 = nx.Graph()
G2.graph['node_dyn'] = True
G2.graph['edge_dyn'] = True


n_nodes = 3

G2.add_node(0)
G2.node[0]['dyn'] = lorenz_node_dyn
G2.node[0]['params'] = [28.0, 10.0, 8.0/3.0]

for i in range(1, n_nodes):
	G2.add_node(i)
	G2.node[i]['dyn'] = lorenz_node_dyn
	G2.node[i]['params'] = [28.0, 10.0, 8.0/3.0]
	G2.add_edge(i-1,i)
	G2.edge[i-1][i]['dyn'] = adaptive_law_edge_dyn
	G2.edge[i-1][i]['state'] = 0.000001

G2.add_edge(n_nodes-1,0)
G2.edge[n_nodes-1][0]['dyn'] = adaptive_law_edge_dyn
G2.edge[n_nodes-1][0]['state'] = 0.000001	
	
netevo.rnd_uniform_node_states (G2, [(0.1, 20.0), (0.1, 20.0), (0.1, 20.0)])


netevo.simulate_rk45 (G2, 5.0, visual_reporter)
#netevo.simulate_rk45 (G2, 5.0, netevo.state_reporter)
