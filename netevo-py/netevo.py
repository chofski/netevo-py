# ============================================================================
# NetEvo for Python
# Copyright (C) 2012 Thomas E. Gorochowski <tom@chofski.co.uk>
# ---------------------------------------------------------------------------- 
# NetEvo is a computing framework designed to allow researchers to investigate 
# evolutionary aspects of dynamical complex networks. By providing tools to 
# easily integrate each of these factors in a coherent way, it is hoped a 
# greater understanding can be gained of key attributes and features displayed 
# by complex systems.
# 
# NetEvo is open-source software released under the Open Source Initiative 
# (OSI) approved Non-Profit Open Software License ("Non-Profit OSL") 3.0. 
# Detailed information about this licence can be found in the COPYING file 
# included as part of the source distribution.
# 
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ============================================================================


import math
import random
import networkx as nx
import numpy    as np


####################################################
# SIMULATION OF NODE AND EDGE DYNAMICS
####################################################


def simulate_euler (G, t_max, reporter, h=0.01):
	"""
	Simulates the dynamics of a network with continuous time dynamics using a 1st
	order Euler method. Not advised for standard use due to lack of stability.
	Included for comparison to other methods.
	"""
	# Check which types of dynamics exist
	node_dyn = G.graph['node_dyn']
	edge_dyn = G.graph['edge_dyn']
	
	# Inform the reporter of the initial state
	reporter(G, 0.0)
	
	# Cycle through all possible times
	t = 0.0
	while t <= t_max:
		
		# Calculate new state for all nodes and edges
		if node_dyn:
			for n in G.nodes():
				cur_node = G.node[n]
				cur_state = cur_node['state']
				deriv = cur_node['dyn'](G, n, t, cur_state)
				cur_node['new_state'] = cur_state + (h * deriv)
		if edge_dyn:
			for e in G.edges():
				cur_edge = G.edge[e[0]][e[1]]
				cur_state = cur_edge['state']
				deriv = cur_edge['dyn'](G, e[0], e[1], t, cur_state)
				cur_edge['new_state'] = cur_state + (h * deriv)
		
		# Shift state
		if node_dyn:
			for n in G.nodes():
				cur_node = G.node[n]
				cur_node['state'] = cur_node['new_state']
		if edge_dyn:
			for e in G.edges():
				cur_edge = G.edge[e[0]][e[1]]
				cur_edge['state'] = cur_edge['new_state']
		
		# Update t (state is now at this point)
		t += h
		
		# Inform the reporter of the updated state
		reporter(G, t)


def simulate_midpoint (G, t_max, reporter, h=0.01):
	"""
	Simulates the dynamics of a network with continuous time dynamics using a 2nd
	order modified Euler method (mid-point). This has better handling of errors
	than the Euler method, but is also not advised for most systems.
	"""
	# Check which types of dynamics exist
	node_dyn = G.graph['node_dyn']
	edge_dyn = G.graph['edge_dyn']
	
	# Inform the reporter of the initial state
	reporter(G, 0.0)
	
	# Cycle through all possible times
	t = 0.0
	while t <= t_max:
		
		# Calculate new state for all nodes and edges
		if node_dyn:
			for n in G.nodes():
				cur_node = G.node[n]
				cur_state = cur_node['state']
				p1 = (h / 2.0) * cur_node['dyn'](G, n, t, cur_state)
				cur_node['new_state'] = cur_state + (h * cur_node['dyn'](G, n, t + (h / 2.0), cur_state + p1))
		if edge_dyn:
			for e in G.edges():
				cur_edge = G.edge[e[0]][e[1]]
				cur_state = cur_edge['state']
				p1 = (h / 2.0) * cur_edge['dyn'](G, n, t, cur_state)
				cur_edge['new_state'] = cur_state + (h * cur_edge['dyn'](G, n, t + (h / 2.0), cur_state + p1))
		
		# Shift state
		if node_dyn:
			for n in G.nodes():
				cur_node = G.node[n]
				cur_node['state'] = cur_node['new_state']
		if edge_dyn:
			for e in G.edges():
				cur_edge = G.edge[e[0]][e[1]]
				cur_edge['state'] = cur_edge['new_state']
		
		# Update t (state is now at this point)
		t += h
		
		# Inform the reporter of the updated state
		reporter(G, t)


def simulate_rk45 (G, t_max, reporter, h=0.01, adaptive=False, tol=1e-5):
	"""
	Simulates the dynamics of a network with continuous time dynamics using a 4th
	order Runge Kutta approach, specifically the Dormand-Prince method. This is
	the recommended simulator for most cases. This is an explicit method and so is
	not well suited for stiff systems, however, in most cases will work fine with
	sufficiently small time-step.
	"""
	# Check which types of dynamics exist
	node_dyn = G.graph['node_dyn']
	edge_dyn = G.graph['edge_dyn']
	
	# Constants for the calculations
	a21  = (1.0/5.0)
	a31  = (3.0/40.0)
	a32  = (9.0/40.0)
	a41  = (44.0/45.0)
	a42  = (-56.0/15.0)
	a43  = (32.0/9.0)
	a51  = (19372.0/6561.0)
	a52  = (-25360.0/2187.0)
	a53  = (64448.0/6561.0)
	a54  = (-212.0/729.0)
	a61  = (9017.0/3168.0)
	a62  = (-355.0/33.0)
	a63  = (46732.0/5247.0)
	a64  = (49.0/176.0)
	a65  = (-5103.0/18656.0)
	a71  = (35.0/384.0)
	a72  = (0.0)
	a73  = (500.0/1113.0)
	a74  = (125.0/192.0)
	a75  = (-2187.0/6784.0)
	a76  = (11.0/84.0)

	c2   = (1.0 / 5.0)
	c3   = (3.0 / 10.0)
	c4   = (4.0 / 5.0)
	c5   = (8.0 / 9.0)
	c6   = (1.0)
	c7   = (1.0)

	b1   = (35.0/384.0)
	b2   = (0.0)
	b3   = (500.0/1113.0)
	b4   = (125.0/192.0)
	b5   = (-2187.0/6784.0)
	b6   = (11.0/84.0)
	b7   = (0.0)

	b1p  = (5179.0/57600.0)
	b2p  = (0.0)
	b3p  = (7571.0/16695.0)
	b4p  = (393.0/640.0)
	b5p  = (-92097.0/339200.0)
	b6p  = (187.0/2100.0)
	b7p  = (1.0/40.0)
	
	# Inform the reporter of the initial state
	reporter(G, 0.0)
	
	# Cycle through all possible times
	t = h
	while t <= t_max:

		# Calculate new state for all nodes and edges
		if node_dyn:
			for n in G.nodes():
				cur_node = G.node[n]
				cur_state = cur_node['state']
				K1 = cur_node['dyn'](G, n, t, cur_state)
				K2 = cur_node['dyn'](G, n, t + c2*h, cur_state+h*(a21*K1))
				K3 = cur_node['dyn'](G, n, t + c3*h, cur_state+h*(a31*K1+a32*K2))
				K4 = cur_node['dyn'](G, n, t + c4*h, cur_state+h*(a41*K1+a42*K2+a43*K3))
				K5 = cur_node['dyn'](G, n, t + c5*h, cur_state+h*(a51*K1+a52*K2+a53*K3+a54*K4))
				K6 = cur_node['dyn'](G, n, t + h,    cur_state+h*(a61*K1+a62*K2+a63*K3+a64*K4+a65*K5))
				K7 = cur_node['dyn'](G, n, t + h,    cur_state+h*(a71*K1+a72*K2+a73*K3+a74*K4+a75*K5+a76*K6))
				cur_node['new_state'] = cur_state + (h * (b1*K1+b3*K3+b4*K4+b5*K5+b6*K6))
		if edge_dyn:
			for e in G.edges():
				cur_edge = G.edge[e[0]][e[1]]
				cur_state = cur_edge['state']
				K1 = cur_edge['dyn'](G, n, t, cur_state)
				K2 = cur_edge['dyn'](G, n, t + c2*h, cur_state+h*(a21*K1))
				K3 = cur_edge['dyn'](G, n, t + c3*h, cur_state+h*(a31*K1+a32*K2))
				K4 = cur_edge['dyn'](G, n, t + c4*h, cur_state+h*(a41*K1+a42*K2+a43*K3))
				K5 = cur_edge['dyn'](G, n, t + c5*h, cur_state+h*(a51*K1+a52*K2+a53*K3+a54*K4))
				K6 = cur_edge['dyn'](G, n, t + h,    cur_state+h*(a61*K1+a62*K2+a63*K3+a64*K4+a65*K5))
				K7 = cur_edge['dyn'](G, n, t + h,    cur_state+h*(a71*K1+a72*K2+a73*K3+a74*K4+a75*K5+a76*K6))
				cur_edge['new_state'] = cur_state + (h * (b1*K1+b3*K3+b4*K4+b5*K5+b6*K6))
		  
		# Shift state
		if node_dyn:
			for n in G.nodes():
				cur_node = G.node[n]
				cur_node['state'] = cur_node['new_state']
		if edge_dyn:
			for e in G.edges():
				cur_edge = G.edge[e[0]][e[1]]
				cur_edge['state'] = cur_edge['new_state']
		
		# Inform the reporter of the updated state
		reporter(G, t)
		
		# Update t
		t += h


def simulate_steps (G, t_max, reporter):
	"""
	Simulates the dynamics of a network with discrete time dynamics.
	"""
	# Check which types of dynamics exist
	node_dyn = G.graph['node_dyn']
	edge_dyn = G.graph['edge_dyn']
	
	# Inform the reporter of the initial state
	reporter(G, 0)
	
	# Cycle through the steps required
	for t in range(1, t_max+1):
		
		# Calculate new state for all nodes and edges
		if node_dyn:
			for n in G.nodes():
				cur_node = G.node[n]
				cur_node['new_state'] = cur_node['dyn'](G, n, t, cur_node['state'])
		if edge_dyn:
			for e in G.edges():
				cur_edge = G.edge[e[0]][e[1]]
				cur_edge['new_state'] = cur_edge['dyn'](G, e[0], e[1], t, cur_node['state'])
		
		# Shift state
		if node_dyn:
			for n in G.nodes():
				cur_node = G.node[n]
				cur_node['state'] = cur_node['new_state']
		if edge_dyn:
			for e in G.edges():
				cur_edge = G.edge[e[0]][e[1]]
				cur_edge['state'] = cur_edge['new_state']
		
		# Inform the reporter of the updated state
		reporter(G, t)


def state_reporter (G, t):
	"""
	Standard simulation state reporter that outputs the current time and
	node states for the system.
	"""
	output = 't = ' + str(t) + ', state = '
	for i in G.nodes():
		output += str(G.node[i]['state']) + ', '
	print output


def rnd_uniform_node_states (G, state_range):
	"""
	Set the state of each node within the network to a random state in the
	ranges specified in the state_range list. This has the form,
	   state_range = [(1min, 1max), (2min, 2max)...]
	up to the number of states required per node.
	"""
	if len(state_range) == 1:
		r1 = state_range[0][0]
		r2 = state_range[0][1]
		for n in G.nodes():
			G.node[n]['state'] = random.uniform(r1, r2)
	else:
		for n in G.nodes():
			n_state = []
			for s in len(state_range):
				n_state.append(random.uniform(state_range[s][0], state_range[s][0]))
			G.node[n]['state'] = np.array(n_state)


def rnd_uniform_edge_states (G, state_range):
	"""
	Set the state of each edge within the network to a random state in the
	ranges specified in the state_range list. This has the form,
	   state_range = [(1min, 1max), (2min, 2max)...]
	up to the number of states required per edge.
	"""
	if len(state_range) == 1:
		r1 = state_range[0][0]
		r2 = state_range[0][1]
		for e in G.edges():
			G.edge[e[0]][e[1]]['state'] = random.uniform(r1, r2)
	else:
		for e in G.edges():
			e_state = []
			for s in len(state_range):
				e_state.append(random.uniform(state_range[s][0], state_range[s][0]))
			G.edge[e[0]][e[1]]['state'] = np.array(e_state)


####################################################
# EVOLUTION AND OPTIMIZATION
####################################################


