#!/usr/bin/python
"""
STUFF
"""


import netevo

import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def no_node_dyn (G, n, t, state):
	return 0.0


def no_edge_dyn (G, source, target, t, state):
	return 0.0


def kuramoto_node_dyn (G, n, t, state):	
	# Parameters
	cur_params = G.node[n]['params']
	natural_freq = cur_params[0]
	coupling_strength = cur_params[1]
	
	# Calculate the new state value
	sum_coupling = 0.0
	for i in G[n]:
		sum_coupling += math.sin(G.node[i]['state'] - state)
	return math.fmod(state + natural_freq + (coupling_strength * sum_coupling), 6.283)
	

def rossler_node_dyn (G, n, t, state):	
	# Parameters
	cur_params = G.node[n]['params']
	sigma = cur_params[0]
	rho = cur_params[1]
	beta = cur_params[2]

	# Calculate the new state value
	c = np.zeros(np.size(state,0))
	coupling = 0.1
	for i in G[n]:
		c += -coupling * (G.node[i]['state'] - state)
	
	v1 = (sigma    * (state[1] - state[0]))       - c[0]
	v2 = (state[0] * (rho - state[2]) - state[1]) - c[1]
	v3 = (state[0] * state[1] - beta * state[2])  - c[2]
	
	return np.array([v1, v2, v3])


# Test the continuous dynamics

G1 = nx.Graph()
G1.graph['node_dyn'] = True
G1.graph['edge_dyn'] = False

G1.add_node(0)
G1.node[0]['state'] = np.array([2.6, 0.8, 0.9])
G1.node[0]['dyn'] = rossler_node_dyn
G1.node[0]['params'] = [28.0, 10.0, 8.0/3.0]

G1.add_node(1)
G1.node[1]['state'] = np.array([1.2, 2.3, 0.2])
G1.node[1]['dyn'] = rossler_node_dyn
G1.node[1]['params'] = [28.0, 10.0, 8.0/3.0]

G1.add_edge(0,1)

netevo.simulate_rk45 (G1, 2.0, netevo.state_reporter)


# Test the discrete dynamics

fig = plt.figure()

def visual_reporter (G, t):
	"""
	Standard simulation state reporter that outputs the current time and
	node states for the system.
	"""
	plt.clf()
	print t
	pos=nx.circular_layout(G)
	n_sizes = []
	for i in G.nodes():
		n_sizes.append(100.0 * G.node[i]['state'])
	nx.draw(G, pos, node_size=n_sizes)
	fig.canvas.draw()
	#node_color=node_color,
	#with_labels=False)

G2 = nx.Graph()
G2.graph['node_dyn'] = True
G2.graph['edge_dyn'] = True

G2.add_node(0)
G2.node[0]['state'] = 0.1
G2.node[0]['dyn'] = kuramoto_node_dyn
G2.node[0]['params'] = [0.2, 0.1]

G2.add_node(1)
G2.node[1]['state'] = 0.5
G2.node[1]['dyn'] = kuramoto_node_dyn
G2.node[1]['params'] = [0.2, 0.1]

G2.add_edge(0,1)
G2.edge[0][1]['state'] = 0.01
G2.edge[0][1]['dyn'] = no_edge_dyn

netevo.simulate_steps (G2, 100, visual_reporter)
