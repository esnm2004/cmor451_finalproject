import numpy as np
import math 
import scipy as sp
import random
import copy
import networkx as nx

# 100 by 100 puplate with 400 people 
#values are node number and key is dictionary mapping neighbor node to (mean, std)
G = {}

matching = []

threshold = 0


def fw_sum(mu, var):
    num = 0
    denom = 0
    for j in range(len(mu)):
        num += (np.exp(2*mu[j] + var[j])) * (np.exp(var[j])-1)
        denom += np.exp(mu[j] + (var[j]/2))
    sum_var = math.log((num/(denom**2)) + 1)
    sum_mu = math.log(denom) - ((sum_var)/2)
    return (sum_mu, sum_var)

def prob_fast_enough(mu, var, threshold):
    point = ((math.log(threshold)- mu)/math.sqrt(var))
    return sp.stats.norm.cdf(point, 0, 1)

def fw_right_cvar(mu, var, p): #use this for cvar_func in utility call
    return np.exp(mu + (var/2)) * ((1 - sp.stats.norm.cdf(sp.stats.norm.ppf(p) - math.sqrt(var)))/(1-p))

def get_utility(stats, threshold, fast_func, cvar_func, weights):
    p = fast_func(*stats, threshold)
    cvar = cvar_func(*stats, p)
    return weights[0]*p - weights[1]*cvar
    

def propose_matching(matching):

    u_1, u_2 = random.sample(list(matching.keys()), 2)

    new_matching = copy.deepcopy(matching)

    new_matching[u_1] = matching[u_2]
    new_matching[u_2] = matching[u_1]
   
    return new_matching


def get_matching_distr(G, G_nx, matching):
    match_mu = []
    match_var = []
    for pair in matching.keys():
        path = nx.shortest_path(G_nx, pair, matching[pair])
        mu = [G[path[i]][path[i+1]][0] for i in range(0, len(path)-1)]
        var =[G[path[i]][path[i+1]][1] for i in range(0, len(path)-1)]
        (path_mu, path_var) = fw_sum(mu, var)
        match_mu.append(path_mu)
        match_var.append(path_var)
    
    stats = fw_sum(match_mu, match_var)

    return stats
        

def generate_graph(grid_length, num_nodes, decay_factor, sigma_min, sigma_max):
   
    rng = np.random.default_rng(None)
    G = {u:{} for u in range(num_nodes)}
    G_coord = {}

    #av degree should be ~7 for ~400 points

    for u in range(num_nodes):
        coord = np.random.uniform(0, grid_length, 2)
        G_coord[u] = coord

    G_nx = nx.Graph()

    #add using Erdos–Renyi
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            birds_eye = np.linalg.norm(G_coord[u] - G_coord[v], 2)
            if np.random.uniform(0, 1) < np.exp(-birds_eye/(decay_factor* (2*grid_length))): #realistic assumption about road presence. decay factor is between 0 and 1. 1 is extremely unreasonable
                mu = float(np.linalg.norm(G_coord[u] - G_coord[v], 1))
                sigma = float(np.random.uniform(sigma_min, sigma_max))
                G_nx.add_edge(u, v, weight = mu)
                G[u][v] = (mu, sigma**2)
                G[v][u] = (mu, sigma**2)

    components = list(nx.connected_components(G_nx))

    while len(components) > 1:
        u = random.choice(list(components[0]))
        v = random.choice(list(components[1]))
        mu = float(np.linalg.norm(G_coord[u] - G_coord[v], 1))
        G_nx.add_edge(u, v, mu)
        sigma = float(np.uniform(sigma_min, sigma_max))
        G[u][v] = (mu, sigma**2)
        G[v][u] = (mu, sigma**2)

        components = list(nx.connected_components(G_nx))

    return (G, G_nx) #G_nx is not probabilistic but will be used for computing shortest paths
 
def metropolis_hastings(G, G_nx, threshold, fast_func, cvar_func, weights, warmup, iter_count):
    odd_degree_nodes = [node for (node, degree) in G_nx.degree() if degree%2 == 1]
    matching = {}
    num_pairs = int(len(odd_degree_nodes)/2)
    for i in range(num_pairs):
        matching[odd_degree_nodes[i]] = odd_degree_nodes[i + num_pairs]

    matchings = []

    for iter in range(iter_count):
        print(iter)
        stats = get_matching_distr(G, G_nx, matching)
        utility = get_utility(stats, threshold, fast_func, cvar_func, weights)
        new_matching = propose_matching(matching)
        new_stats = get_matching_distr(G, G_nx, new_matching)
        new_utility = get_utility(new_stats, threshold, fast_func, cvar_func, weights)
        if np.random.uniform(0, 1) < np.exp(new_utility-utility):
            matching = new_matching

        if iter> warmup:
            matchings.append(matching)

    return matchings
        
(G, G_nx) = generate_graph(100, 400, 0.1, 0.3, 0.5)

matchings = metropolis_hastings(G, G_nx, 10000, prob_fast_enough, fw_right_cvar, [1, 0.01], 100, 400)

print(matchings)

