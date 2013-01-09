"""
NetEvo for Python
=================

    NetEvo is a computing framework designed to allow researchers to 
    investigate evolutionary aspects of dynamical complex networks. It 
    provides functionality to easily simulate dynamical networks with both 
    nodes and edges states, and includes optimization methods to evolve
    the dynamics or structure of a system towards some user specified goal.
    
    NetEvo is writen in Python and makes use of the networkx, numpy, and SciPy
    packages.
"""
#    NetEvo for Python
#    Copyright (C) 2010-2013 by
#    Thomas E. Gorochowski <tom@chofski.co.uk>
#    All rights reserved.
#    OSI Non-Profit Open Software License ("Non-Profit OSL") 3.0 license.

import sys
if sys.version_info[:2] < (2, 6):
    m = "Python version 2.6 or later is required for NetEvo (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
del sys

__author__  = 'Thomas E. Gorochowski <tom@chofski.co.uk>'
__license__ = 'OSI Non-Profit OSL 3.0'
__version__ = '1.0'

import math
import random
import pickle
import networkx as nx
import numpy    as np
import scipy.integrate as integrate

def simulate_euler(G, t_max, reporter=None, h=0.01):
    """Simulate continuous-time network dynamics using a 1st order Euler 
    method.
    
    This method is very simple and not advised for general use. It is included
    for comparison and teaching purposes. The state of the simulaton is
    stored as a node or edge attribute with the 'state' key.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    t_max :  float
        Time to simulate for.
    
    reporter : function (optional default=None)
        Reporting function called at each timestep, see: state_reporter(G, t).
    
    h : float (default=0.01)
        Timestep
    """
    # Check which types of dynamics exist
    node_dyn = G.graph['node_dyn']
    edge_dyn = G.graph['edge_dyn']
    # Inform the reporter of the initial state
    if reporter != None:
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
                deriv = cur_edge['dyn'](G, e, t, cur_state)
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
        if reporter != None:
            reporter(G, t)

def simulate_midpoint(G, t_max, reporter=None, h=0.01):
    """Simulate continuous-time network dynamics using a 2nd order modified 
    Euler method (mid-point).
    
    This has better handling of errors than the 1st order Euler method, but is
    also not advised for most systems. It is included for comparison and 
    teaching purposes. The state of the simulaton is stored as a node or edge
    attribute with the 'state' key.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    t_max :  float
        Time to simulate for.
    
    reporter : function (optional default=None)
        Reporting function called at each timestep, see: state_reporter(G, t).
    
    h : float (default=0.01)
        Timestep
    """
    # Check which types of dynamics exist
    node_dyn = G.graph['node_dyn']
    edge_dyn = G.graph['edge_dyn']
    # Inform the reporter of the initial state
    if reporter != None:
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
                cur_node['new_state'] = cur_state + (h * cur_node['dyn'](G, n,
                                               t + (h / 2.0), cur_state + p1))
        if edge_dyn:
            for e in G.edges():
                cur_edge = G.edge[e[0]][e[1]]
                cur_state = cur_edge['state']
                p1 = (h / 2.0) * cur_edge['dyn'](G, e, t, cur_state)
                cur_edge['new_state'] = cur_state + (h * cur_edge['dyn'](G, n,
                                               t + (h / 2.0), cur_state + p1))
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
        if reporter != None:
            reporter(G, t)

def simulate_rk45(G, t_max, reporter=None, h=0.01):
    """Simulate continuous-time network dynamics using a 4th order Runge Kutta 
    method (Dormand-Prince).
    
    This is the recommended simulator for most cases. It is an explicit method
    and so is not always well suited for stiff systems, however, in most cases
    it is suitable with a sufficiently small timestep. The state of the 
    simulaton is stored as a node or edge attribute with the 'state' key.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    t_max :  float
        Time to simulate for.
    
    reporter : function (optional default=None)
        Reporting function called at each timestep, see: state_reporter(G, t).
    
    h : float (default=0.01)
        Timestep
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
    if reporter != None:
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
                K3 = cur_node['dyn'](G, n, t + c3*h, cur_state+h*(a31*K1+a32*
                                                                  K2))
                K4 = cur_node['dyn'](G, n, t + c4*h, cur_state+h*(a41*K1+a42*
                                                                  K2+a43*K3))
                K5 = cur_node['dyn'](G, n, t + c5*h, cur_state+h*(a51*K1+a52*
                                                            K2+a53*K3+a54*K4))
                K6 = cur_node['dyn'](G, n, t + h, cur_state+h*(a61*K1+a62*K2+
                                                        a63*K3+a64*K4+a65*K5))
                K7 = cur_node['dyn'](G, n, t + h, cur_state+h*(a71*K1+a72*K2+
                                                 a73*K3+a74*K4+a75*K5+a76*K6))
                cur_node['new_state'] = cur_state + (h * (b1*K1+b3*K3+b4*K4+
                                                          b5*K5+b6*K6))
        if edge_dyn:
            for e in G.edges():
                cur_edge = G.edge[e[0]][e[1]]
                cur_state = cur_edge['state']
                K1 = cur_edge['dyn'](G, e, t, cur_state)
                K2 = cur_edge['dyn'](G, e, t + c2*h, cur_state+h*(a21*K1))
                K3 = cur_edge['dyn'](G, e, t + c3*h, cur_state+h*(a31*K1+a32*
                                                                  K2))
                K4 = cur_edge['dyn'](G, e, t + c4*h, cur_state+h*(a41*K1+a42*
                                                                  K2+a43*K3))
                K5 = cur_edge['dyn'](G, e, t + c5*h, cur_state+h*(a51*K1+a52*
                                                            K2+a53*K3+a54*K4))
                K6 = cur_edge['dyn'](G, e, t + h, cur_state+h*(a61*K1+a62*K2+
                                                        a63*K3+a64*K4+a65*K5))
                K7 = cur_edge['dyn'](G, e, t + h, cur_state+h*(a71*K1+a72*K2+
                                                 a73*K3+a74*K4+a75*K5+a76*K6))
                cur_edge['new_state'] = cur_state + (h * (b1*K1+b3*K3+b4*K4+
                                                                 b5*K5+b6*K6))
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
        if reporter != None:
            reporter(G, t)
        # Update t
        t += h

def simulate_ode_fixed(G, ts, node_dim=1, edge_dim=1, rtol=1e-5, atol=1e-5, 
                      save_final_state=True):
    """Simulate continuous-time network dynamics using the SciPy odeint 
    function (adaptive step integrator).
    
    For systems where simulation does not lead to a change in the network
    structure and where node and edge states maintain the same size through
    time, it is possible to use the built-in SciPy ode solvers. Note special
    dynamic functions for nodes and edges must be used. Initial condition is
    defined in the 'state' attribute of nodes and edges in G.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    ts :  list(float)
        List of time points to output the simulation results.
        
    node_dim : int (default=1)
        The dimension of node states.
    
    edge_dim : int (default=1)
        The dimension of edge states.
    
    rtol : float (default=1e-5)
        Relative error tolerance to be maintained (passed to SciPy).
    
    ratol : float (default=1e-5)
        Absolute error tolerance to be maintained (passed to SciPy).
    
    save_final_state : boolean (default=True) 
        Flag to choose if the final simulation state should be saved to the
        networks 'state' attribute for the associated nodes and edges.
    
    Returns
    -------
    res: numpy.array
        Array of the simulation results. A row exists for each of the given
        timepoints in ts and columns represent the node and edge states. To
        find the approriate starting index for a particular node or edge the
        returned mappings must be used.
    
    nmap: dict
        A dictionary keyed by the node. Returns the position in the results
        array (res) of the first state value for that node.
    
    emap: dict
        A dictionary keyed by the edge. Returns the position in the results
        array (res) of the first state value for that edge.
    """    
    # Generate the node and edge mappings for the state vector
    nmap = {}
    emap = {}
    max_node_idx = 0
    # Create the node mapping
    if G.graph['node_dyn'] == True:
        for idx, n in enumerate(G.nodes()):
            nmap[n] = idx * node_dim
            max_node_idx = node_dim * G.number_of_nodes()
    else:
        nmap = None
        node_dim = 0
    # Create the edge mapping
    if G.graph['edge_dyn'] == True:
        for idx, e in enumerate(G.edges()):
            emap[e] = max_node_idx + (idx * edge_dim)
    else:
        emap = None
        edge_dim = 0
    # Generate the initial conditions (from G 'state')
    f0 = np.zeros(max_node_idx + (G.number_of_edges() * edge_dim))
    if nmap != None:
        for n in G.nodes():
            state = G.node[n]['state']
            f0[nmap[n]:(nmap[n] + node_dim)] = state
    if emap != None:
        for e in G.edges():
            state = G.edge[e[0]][e[1]]['state']
            f0[emap[e]:(emap[e] + edge_dim)] = state
    # Simulate the system
    res = integrate.odeint(simulate_ode_fixed_fn, f0, ts, args=(G, nmap, 
                           emap), rtol=rtol, atol=atol)
    # Save the final state to G
    if save_final_state:
        if nmap != None:
            for n in G.nodes():
                G.node[n]['state'] = res[:][-1][nmap[n]:(nmap[n] + node_dim)]
        if emap != None:
            for e in G.edges():
                G.edge[e[0]][e[1]]['state'] = res[:][-1][emap[e]:(emap[e] + 
                                                                  edge_dim)]
    # Return the full simulation array
    return res, nmap, emap

def simulate_ode_fixed_fn(y, t, G, nmap, emap):
    # Internal function for calculating network derivitive
    dy = np.zeros(len(y))
    if nmap != None:
        # Call all the node update functions
        for n in G.nodes():
            G.node[n]['dyn'](G, n, t, y, dy, nmap, emap)
    if emap != None:
        # Call all the edge update functions
        for e in G.edges():
            G.edge[e[0]][e[1]]['dyn'](G, e, t, y, dy, nmap, emap)
    return dy

def simulate_steps(G, t_max, reporter=None):
    """Simulate discrete-time network dynamics.
    
    This is the recommended simulator for most cases. The state of the 
    simulaton is stored as a node or edge attribute with the 'state' key.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    t_max :  float
        Time to simulate for.
    
    reporter : function (optional default=None)
        Reporting function called at each timestep, see: state_reporter(G, t).
    """
    # Check which types of dynamics exist
    node_dyn = G.graph['node_dyn']
    edge_dyn = G.graph['edge_dyn']
    # Inform the reporter of the initial state
    if reporter != None:
        reporter(G, 0)
    # Cycle through the steps required
    for t in range(1, t_max+1):
        # Calculate new state for all nodes and edges
        if node_dyn:
            for n in G.nodes():
                cur_node = G.node[n]
                cur_node['new_state'] = cur_node['dyn'](G, n, t, 
                                                        cur_node['state'])
        if edge_dyn:
            for e in G.edges():
                cur_edge = G.edge[e[0]][e[1]]
                cur_edge['new_state'] = cur_edge['dyn'](G, e, t, 
                                                        cur_node['state'])
        
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
        if reporter != None:
            reporter(G, t)

def simulate_steps_fixed(G, ts, node_dim=1, edge_dim=1, 
                         save_final_state=True):
    """Simulate discrete-time network dynamics.
    
    For systems where simulation does not lead to a change in the network
    structure and where node and edge states maintain the same size through
    time. Note special dynamic functions for nodes and edges must be used. 
    Initial condition is defined in the 'state' attribute of nodes and edges
    in G.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    ts :  list(float)
        List of time points to output the simulation results.
        
    node_dim : int (default=1)
        The dimension of node states.
    
    edge_dim : int (default=1)
        The dimension of edge states.
    
    save_final_state : boolean (default=True) 
        Flag to choose if the final simulation state should be saved to the
        networks 'state' attribute for the associated nodes and edges.
    
    Returns
    -------
    res: numpy.array
        Array of the simulation results. A row exists for each of the given
        timepoints in ts and columns represent the node and edge states. To
        find the approriate starting index for a particular node or edge the
        returned mappings must be used.
    
    nmap: dict
        A dictionary keyed by the node. Returns the position in the results
        array (res) of the first state value for that node.
    
    emap: dict
        A dictionary keyed by the edge. Returns the position in the results
        array (res) of the first state value for that edge.
    """
    # Check which types of dynamics exist
    node_dyn = G.graph['node_dyn']
    edge_dyn = G.graph['edge_dyn']
    # Variable to hold the results
    res = []
    # Generate the node and edge mappings for the state vector
    nmap = {}
    emap = {}
    max_node_idx = 0
    # Create the node mapping
    if G.graph['node_dyn'] == True:
        for idx, n in enumerate(G.nodes()):
            nmap[n] = idx * node_dim
            max_node_idx = node_dim * G.number_of_nodes()
    else:
        nmap = None
        node_dim = 0
    # Create the edge mapping
    if G.graph['edge_dyn'] == True:
        for idx, e in enumerate(G.edges()):
            emap[e] = max_node_idx + (idx * edge_dim)
    else:
        emap = None
        edge_dim = 0
    # Generate the initial conditions (from G 'state')
    y = np.zeros(max_node_idx + (G.number_of_edges() * edge_dim))
    if nmap != None:
        for n in G.nodes():
            y[nmap[n]:(nmap[n] + node_dim)] = G.node[n]['state']
    if emap != None:
        for e in G.edges():
            y[emap[e]:(emap[e] + edge_dim)] = G.edge[e[0]][e[1]]['state']
    # Save the initial conditions
    res.append(y)
    # Cycle through the steps required
    for t in range(1, max(ts)+1):
        # Create a new state vector
        dy = np.zeros(len(y))
        if nmap != None:
            # Call all the node update functions
            for n in G.nodes():
                G.node[n]['dyn'](G, n, t, y, dy, nmap, emap)
        if emap != None:
            # Call all the edge update functions
            for e in G.edges():
                G.edge[e[0]][e[1]]['dyn'](G, e, t, y, dy, nmap, emap)
        # Save the state if in the output list
        if t in ts:
            res.append(dy)
        y = dy
    # Save the final state to G
    if save_final_state:
        if nmap != None:
            for n in G.nodes():
                G.node[n]['state'] = res[:][-1][nmap[n]:(nmap[n] + node_dim)]
        if emap != None:
            for e in G.edges():
                G.edge[e[0]][e[1]]['state'] = res[:][-1][emap[e]:(emap[e] + 
                                                                  edge_dim)]
    return np.array(res), nmap, emap

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
            for s in range(len(state_range)):
                n_state.append(random.uniform(state_range[s][0], 
                                              state_range[s][1]))
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
            for s in range(len(state_range)):
                e_state.append(random.uniform(state_range[s][0], 
                                              state_range[s][1]))
            G.edge[e[0]][e[1]]['state'] = np.array(e_state)


def set_all_node_dynamics (G, dyn_fn):
    for n in G.nodes():
        G.node[n]['dyn'] = dyn_fn


def set_all_edge_dynamics (G, dyn_fn):
    for e in G.edges():
        G.edge[e[0]][e[1]]['dyn'] = dyn_fn


####################################################
# DYNAMICS LIBRARY
####################################################


def no_node_dyn (G, n, t, state):
    """
    No node dynamics to be used as a null form of node dynamics.
    """
    return 0.0


def no_edge_dyn (G, source, target, t, state):
    """
    No edge dynamics to be used as a null form of edge dynamics.
    """
    return 0.0


####################################################
# EVOLUTION AND OPTIMIZATION
####################################################


def random_rewire (G, n):
    nodes = G.nodes()
    edges = G.edges()
    for i in range(n):
        # Pick a random edge
        (u, v) = edges[int(random.random()*G.number_of_edges())]
        # Attempt to find a new random edge (maximum 1000 trials)
        trial = 0
        while trial < 1000:
            new_u = int(random.random()*len(G))
            new_v = int(random.random()*len(G))
            if new_u != new_v and \
               G.has_edge(nodes[new_u], nodes[new_v]) == False:
                break
            trial += 1  
        # Rewire if max trials not reached
        if trial >= 1000:
            print 'WARNING: Could not rewire edge - max trials exceeded'
        else:
            # Rewire it
            G.remove_edge(u, v)
            G.add_edge(nodes[new_u], nodes[new_v])


def evo_sa_reporter (G, G_perf, iteration):
    print 'Iteration: ', iteration, ', Performance = ', G_perf


def boltzmann_accept_prob (d_perf, temperature):
    return math.exp(d_perf / temperature);


def evolve_sa (G, perf_fn, mut_fn, max_iter=100000, max_no_change=100, 
               initial_temp=100000000000.0, min_temp=0.001, 
               reporter=evo_sa_reporter, cooling_rate=0.99, 
               accept_prob_fn=boltzmann_accept_prob):
    """
    Evolves a network using a simulated annealing metaheuristic.
    """
    # Copy the system and set initial process variables
    cur_G = G.copy()
    iteration = 0
    cur_temp = initial_temp
    
    # Calculate the initial performance
    cur_perf = perf_fn(cur_G)
        
    # Observe the inital system
    reporter(cur_G, cur_perf, iteration)

    no_change = 0   
    if cur_temp > 0.0:
        while no_change <= max_no_change and cur_temp > min_temp and \
              iteration <= max_iter:
            iteration += 1
            # Run a trial
            accept, new_G, G_perf = evolve_sa_trial(cur_temp, cur_perf, 
                                       cur_G, mut_fn, perf_fn, accept_prob_fn)
            if accept:
                cur_G = new_G
                cur_perf = G_perf
                no_change = 0
            else:
                no_change += 1
            # Observe the current system
            reporter(cur_G, cur_perf, iteration)
            # Reduce the temperature
            cur_temp *= cooling_rate            
    else:
        print 'WARNING: Initial temperature was <= 0.0'
    
    return iteration, cur_G


def evolve_sa_trial (cur_temp, cur_perf, G, mut_fn, perf_fn, accept_prob_fn):
    # Make a copy of the system
    G_copy = G.copy()
    
    # Mutate the system
    mut_fn(G_copy)
    
    # Estimate performance
    new_perf = perf_fn(G_copy)
    
    if new_perf == float('inf'):
        # Do not accept change
        return False, G, cur_perf
    
    d_perf =  cur_perf - new_perf
    if d_perf > 0.0:
        # Accept improvement
        return True, G_copy, new_perf
    else:
        # Ensure positive temperature
        if cur_temp > 0.0:
            # Randomly accept in relation to temperature
            if random.random() <= accept_prob_fn(d_perf, cur_temp):
                return True, G_copy, new_perf
        else:
            print 'WARNING: Zero or negative temperature (evolve_sa_trail)'
    
    # Mutation not accepted
    return False, G, cur_perf


def evo_ga_reporter (G_pop, G_pop_perf, iteration):
    print 'Iteration: ', iteration, ', Performance = '


def evolve_ga(G_pop, perf_fn, repoduce_fn, max_iter=10000,
              reporter=evo_ga_reporter):
    """
    Evolves a population of networks using a genetic algorithm metaheuristic. Runs 
    each simulation step as a separate process to make use of multi-processor systems.
    """
    print 'TODO'


def graph_random_mutate (G, node_add_prob=0.0, node_del_prob=0.0, 
                         edge_rewire_prob=0.0, edge_add_prob=0.0, 
                         edge_del_prob=0.0):
    """
    Mutate in place - don't create a new object
    """
    print 'TODO'


def graph_crossover (G1, G2, points=1):
    """
    Returns a new graph object (deepcopy) containing the crossed over graph
    """
    # Pick n random numbers and sort - these are the crossover points
    print 'TODO'


def find_differences (G1, G2):
    """Find the differences between two graphs and output a string in the
    NetEvoX format (line per change). This is prodominantly of use for 
    visualisation of evolving topologies over time.
    """
    print 'TODO'

def write_to_file (G, path, format='gml', node_keys=[], edge_keys=[]):
    """Writes a netevo graph to a suitably formatted GraphML file for use in 
    external applications such as Cytoscape. This should be used instead of
    the networkx functions as Cytoscape does not correctly handle non-string
    based labels or lists (often used for parameters).
    """
    G_copy = G.copy()
    for n in G_copy.nodes():
        G_copy.node[n]['label'] = str(n)
        if G_copy.graph['node_dyn'] == True:
            G_copy.node[n]['dyn'] = str(G_copy.node[n]['dyn'])
        for k in node_keys:
            G_copy.node[n][k] = str(G_copy.node[n][k])
    for n in G_copy.edges():
        if G_copy.graph['edge_dyn'] == True:
            G_copy.edge[e[0]][e[1]]['dyn']=str(G_copy.edge[e[0]][e[1]]['dyn'])
        for k in edge_keys:
            G_copy.edge[e[0]][e[1]][k] = str(G_copy.edge[e[0]][e[1]][k])
    if format == 'gml':
        nx.write_gml(G_copy, path)
    elif format == 'graphml':
        nx.write_graphml(G_copy, path)
    else:
        print 'WARNING: Unsupported file format (', format, ')'
