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
import networkx as nx
import numpy as np
import scipy.integrate as integrate

def simulate_euler (G, t_max, reporter=None, h=0.01):
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

def simulate_midpoint (G, t_max, reporter=None, h=0.01):
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

def simulate_rk45 (G, t_max, reporter=None, h=0.01):
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

def simulate_ode_fixed (G, ts, node_dim=1, edge_dim=1, rtol=1e-5, atol=1e-5, 
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

def simulate_ode_fixed_fn (y, t, G, nmap, emap):
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

def simulate_steps (G, t_max, reporter=None):
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

def simulate_steps_fixed (G, ts, node_dim=1, edge_dim=1, 
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
    """Simple simulation state reporter that outputs the current time and
    node states.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    t :  float
        Time point of the simulation.
    """
    output = 't = ' + str(t) + ', state = '
    for i in G.nodes():
        output += str(G.node[i]['state']) + ', '
    print output

def rnd_uniform_node_states (G, state_range):
    """Set all node states in a network to a uniformly random value.
    
    To allow for states of dimension > 1, state ranges should be provided for
    each element in the state vector.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    state_range :  list(tuples)
        List of tuples that hold the min and max value to randomly pick a 
        value between e.g., state_range = [(1min, 1max), (2min, 2max)...].
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
    """Set all edge states in a network to a uniformly random value.
    
    To allow for states of dimension > 1, state ranges should be provided for
    each element in the state vector.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    state_range :  list(tuples)
        List of tuples that hold the min and max value to randomly pick a 
        value between e.g., state_range = [(1min, 1max), (2min, 2max)...].
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
    """Set the dynamics for all nodes.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    dyn_fn :  function
        Function to be used for every nodes dynamics.
    """
    for n in G.nodes():
        G.node[n]['dyn'] = dyn_fn

def set_all_edge_dynamics (G, dyn_fn):
    """Set the dynamics for all edges.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    dyn_fn :  function
        Function to be used for every edges dynamics.
    """
    for e in G.edges():
        G.edge[e[0]][e[1]]['dyn'] = dyn_fn

def no_node_dyn (G, n, t, state):
    """Null node dynamics (does nothing).

    To be used when you want some nodes to have no dynamics.
    """
    return 0.0

def no_edge_dyn (G, e, t, state):
    """Null edge dynamics (does nothing).

    To be used when you want some edges to have no dynamics.
    """
    return 0.0

def random_rewire (G, n, allow_self_loops=False):
    """Randomly rewire edges.

    This function performs a full rewire i.e., it will ensure the newly created
    edge contains all the same properties as the original.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    n :  int
        Number of edges to randomly rewire.
        
    allow_self_loops : boolean (default=False)
        Flag as to whether self loops are allowed.
    """
    nodes = G.nodes()
    edges = G.edges()
    for i in range(n):
        # Pick a random edge
        (u, v) = edges[int(random.random()*G.number_of_edges())-1]
        # Attempt to find a new random edge (maximum 1000 trials)
        trial = 0
        while trial < 1000:
            new_u = int(random.random()*len(G))
            new_v = int(random.random()*len(G))
            if allow_self_loops:
                if G.has_edge(nodes[new_u], nodes[new_v]) == False:
                    break
            else:
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
    """Simple evolutionary state reporter for the simulated annealing evolver.
    
    Outputs the current iteration and performance value for the network.
    
    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).

    G_perf :  float
        Performance of the network.
        
    iteration : int
        Iteration of the evolutionary process.
    """
    print 'Iteration: ' + str(iteration) + ', Performance = ' + str(G_perf)

def boltzmann_accept_prob (d_perf, temperature):
    """Boltzmann accepting probability function for the simulated annealing
    evolver.
    
    Parameters
    ----------
    d_perf : float
        Change in performance value in last iteration.

    temperature :  float
        Current temperature of the simulated annealing process.
    """
    return math.exp(d_perf / temperature);

def evolve_sa (G, perf_fn, mut_fn, max_iter=100000, max_no_change=100, 
               initial_temp=100000000000.0, min_temp=0.001, 
               reporter=None, cooling_rate=0.99, 
               accept_prob_fn=boltzmann_accept_prob):
    """Simulated annealing based evolver.
    
    Starting wit 
    
    Parameters
    ----------
    G : NetworkX graph
        Starting network to evolve. It is assumed that this is configured for 
        use with NetEvo, with defined dynamics for each node or edge
        (as appropriate).
        
    perf_fn : function
        Performance function to evalulate each candidate network. Lower 
        performance values are better - evolution minimizes.
        
    mut_fn : function
        Mutation function to generate new candidate networks from an existing
        network.
    
    max_iter : int (default=100000)
        Maximum number of iterations to perform.
    
    max_no_change : int (default=100)
        Maximum number of consecutive iterations with no change before 
        halting.
    
    initial_temp : float (default=100000000000.0)
        Initial temperature of the simulated annealing process.
    
    min_temp : float (default=0.001)
        Minimum temperature of the simulated annealing process before halting.
    
    reporter : function (optional default=None)
        Optional reporter called after each evolutionary step.
    
    cooling_rate : float (default=0.99)
        The fraction of the temperature used in following iterations.
    
    accept_prob_fn : function (default=boltzmann_accept_prob)
        Function defining the accepting probability at a particular 
        temperature.
        
    Returns
    -------
    iteration : int
        Final iteration reached
        
    cur_G : NetworkX graph
        Resultant network from the evolutionary process
    """
    # Copy the system and set initial process variables
    cur_G = G.copy()
    iteration = 0
    cur_temp = initial_temp
    # Calculate the initial performance
    cur_perf = perf_fn(cur_G)
    # Observe the inital system
    if reporter != None:
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
            if reporter != None:
                reporter(cur_G, cur_perf, iteration)
            # Reduce the temperature
            cur_temp *= cooling_rate            
    else:
        print 'WARNING: Initial temperature was <= 0.0'
    return iteration, cur_G

def evolve_sa_trial (cur_temp, cur_perf, G, mut_fn, perf_fn, accept_prob_fn):
    # Internal function that calculates a simulated annealing trial
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

def evo_ga_reporter (G_pop_perf, iteration):
    """Simple evolutionary state reporter for the genetic algorithms evolver.
    
    Outputs the current iteration and performance values for the network
    populations.
    
    Parameters
    ----------
    G_pop_perf : list([NetworkX graph, float])
        Current evolving network population with the performance value.
        
    iteration : int
        Iteration of the evolutionary process.
    """
    out_str = 'Iteration: ' + str(iteration) + ', Performance = '
    for perf in G_pop_perf:
        out_str += str(perf[1]) + ', ' 
    print out_str

def evolve_ga (G_pop, perf_fn, reproduce_fn, max_iter=1000,
              reporter=None):
    """ Evolves a population of networks using a genetic algorithm.

    Outputs the evolved population with the accociated performance values.

    Parameters
    ----------
    G_pop : list(NetworkX graph)
        Initial network population.
    
    perf_fn : function
        Performance function to evalulate each candidate network. Lower 
        performance values are better - evolution minimizes.
        
    reproduce_fn : function
        Function to generate new candidate networks from an existing
        population with performance values.

    max_iter : int (default = 1000)
        Maximum number of iterations (generations) to produce.

    reporter : function (optional default=None)
        Optional reporter called after each evolutionary step.
    """
    # Copy the population (we don't make changes to the initial one)
    cur_pop_perf = []
    for g in G_pop:
        cur_pop_perf.append([g, 0.0])
    for it in range(0, max_iter):
        # Calculate the performance
        perf_fn(cur_pop_perf)
        # Report the current performance
        if reporter != None:
            reporter(cur_pop_perf, it)
        # Mate the graphs and update the current population
        cur_pop_perf = reproduce_fn(cur_pop_perf)
    # Report the final performance
    if reporter != None:
        reporter(cur_pop_perf, max_iter)
    return cur_pop_perf

def evolve_ga_reproduce (G_pop_perf, n_dup_prob=0.02, n_del_prob=0.02,
                         e_dup_prob=0.02, e_del_prob=0.02, points=1):
    """ A basic reproduce function that will randomly duplicate and delete 
    nodes and edges, and perform network crossover on a population of networks 
    to generate a new candidate population for the genetic algorithm.

    Can be used with default values or called from a user defined
    function that specifies particular probabilities and crossover points to 
    use. Due to the reproduction often being highly constrainted in natural and
    engineered systems, we recommend creating custom versions for the specific
    system being studied.

    Outputs the new candidate population set (all performance values set to 0).

    Parameters
    ----------
    G_pop_perf : list([NetworkX graph, float])
        Current evolving network population with the performance value.

    n_dup_prob : float (default = 0.02)
        Node duplication probability.

    n_del_prob : float (default = 0.02)
        Node deletion probability.

    e_dup_prob : float (default = 0.02)
        Edge duplication probability.

    e_del_prob : float (default = 0.02)
        Edge deletion probability.
    
    points : int (default = 1)
        Number of crossover points.
    """
    print 'WARNING: Currently not implemented.'


def graph_crossover (G1, G2, points=1):
    """ Performs a network based crossover operation on two graphs.

    Outputs the crossovered graph (new object).

    Parameters
    ----------
    G1 : NetworkX graph
        Graph 1 to crossover.

    G2 : NetworkX graph
        Graph 2 to crossover.
    
    points : int (default = 1)
        Number of crossover points.
    """
    # Pick a node number of perform the crossover with
    nodes_1 = G1.nodes()
    nodes_2 = G2.nodes()
    # Randomly choose crossover points (should include check that loop will end)
    if points >= G1.number_of_nodes():
        print 'ERROR: Too many crossover points (defaulting to 1).'
        points = 1
    n_cross_points = [0]
    for p in range(points):
        new_p = int(random.random()*G1.number_of_nodes())
        while new_p not in n_cross_points:
            new_p = int(random.random()*G1.number_of_nodes())
        n_cross_points.append(new_p)
    n_cross_points = sorted(n_cross_points)
    # Sets of nodes to extract for each graph
    g_num = 1
    ns_1 = []
    ns_2 = []
    for p_idx in range(1,len(n_cross_points)):
        p1_idx = n_cross_points[p_idx-1]
        p2_idx = n_cross_points[p_idx]
        if g_num == 1:
            ns_1 += nodes_1[p1_idx:p2_idx]
            g_num = 2
        else:
            ns_2 += nodes_2[p1_idx:p2_idx]
            g_num = 1
    # Handle the case where both lists might include the same nodes (clean up)
    for i in ns_2:
        if i in ns_1:
            # Remove node from list 2
            ns_2.remove(i)
    # Generate new network that is a crossover of the two
    G_cross = nx.create_empty_copy(G1)
    # Copy graph properties
    for k in G1.graph.keys():
        G_cross.graph[k] = G1.graph[k]
    # Remove all nodes not in ns_1 list
    for n in ns_1:
        G_cross.add_node(n)
        # Copy all properties from G1
        g1_n = G1.node[n]
        g1_n_keys = g1_n.keys()
        for k in g1_n_keys:
            G_cross.node[n][k] = g1_n[k]
    # Add all nodes from ns_2
    for n in ns_2:
        G_cross.add_node(n)
        # Copy all properties from G2
        g2_n = G2.node[n]
        g2_n_keys = g2_n.keys()
        for k in g2_n_keys:
            G_cross.node[n][k] = g2_n[k]
    # Add edges present where nodes still exist in crossovered graph
    #for n in ns_1:
        # Check that source and target in new graph, if so add with properties
        # TODO
    #    pass
    #for n in ns_2:
        # TODO
    #    pass
    return G_cross

def write_to_file (G, path, format='gml', node_keys=[], edge_keys=[]):
    """Writes a NetEvo graph to a suitably formatted file for use in 
    external applications such as Cytoscape.
    
    This should be used instead of the networkx functions as Cytoscape does 
    not correctly handle non-string based labels or lists (often used for 
    parameters). Parameters to convert can be specified.
    
    Outputs a file in the designated format.

    Parameters
    ----------
    G : NetworkX graph
        It is assumed that this is configured for use with NetEvo, with 
        defined dynamics for each node or edge (as appropriate).
        
    path : string
        Filename and path of the output file.
        
    format : string "gml"|"graphml" (default="gml")
        Output format.
        
    node_keys : list(string)
        List of node attribute keys to convert to strings.
    
    edge_keys : list(string)
        List of edge attribute keys to convert to strings.
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
