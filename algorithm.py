import numpy as np
import math 
import scipy as sp
import random
import copy
import networkx as nx
import time

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

def fw_right_cvar(mu, var, p):  #use this for cvar_func in utility call
    if p == 1:
        print("probability was 1")
        return 0
    else:
        return np.exp(mu + (var/2)) * ((1 - sp.stats.norm.cdf(sp.stats.norm.ppf(p) - math.sqrt(var)))/(1-p))

def get_utility(stats, threshold, fast_func, cvar_func, weights):
    p = fast_func(*stats, threshold)
    cvar = cvar_func(*stats, p)
    # print(cvar)
    return weights[0]*p - weights[1]*(cvar/threshold)
    

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
   
    rng = np.random.default_rng(3)
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
                sigma = float(np.random.uniform(sigma_min, sigma_max))
                mu = math.log(float(np.linalg.norm(G_coord[u] - G_coord[v], 1))) - (sigma/2) #choose mu so that the mean travel time is the taxicab distance
                G_nx.add_edge(u, v, weight = mu)
                G[u][v] = (mu, sigma**2)
                G[v][u] = (mu, sigma**2)

    components = list(nx.connected_components(G_nx))

    while len(components) > 1:
        u = random.choice(list(components[0]))
        v = random.choice(list(components[1]))
        mu = float(np.linalg.norm(G_coord[u] - G_coord[v], 1))
        G_nx.add_edge(u, v, weight=mu)                                            # Fixed bug, didn't have weight=mu
        sigma = float(np.random.uniform(sigma_min, sigma_max))                    # Fixed bug, didn't have .random
        G[u][v] = (mu, sigma**2)
        G[v][u] = (mu, sigma**2)

        components = list(nx.connected_components(G_nx))

    return (G, G_nx) #G_nx is not probabilistic but will be used for computing shortest paths
 
def metropolis_hastings(G, G_nx, threshold, fast_func, cvar_func, weights, warmup, iter_count):
    odd_degree_nodes = [node for (node, degree) in G_nx.degree() if degree%2 == 1]
    matching = {}
    num_pairs = int(len(odd_degree_nodes)/2)
    print(num_pairs)
    for i in range(num_pairs):
        matching[odd_degree_nodes[i]] = odd_degree_nodes[i + num_pairs]

    matchings = []

    for iter in range(iter_count):
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
        

def test_weights(G, G_nx, weights, warmup, itercount, threshold):

    
    start_time = time.time()
    matchings = metropolis_hastings(G, G_nx, threshold, prob_fast_enough, fw_right_cvar, weights, warmup, itercount)
    end_time = time.time()
    seen = set()
    distinct_matchings = []
    for matching in matchings:
        t = tuple(sorted(matching.items()))
        if t not in seen:
            seen.add(t)
            distinct_matchings.append(matching)
    num_matchings = len(distinct_matchings)
    run_time = end_time-start_time
    avg_run_time = run_time/num_matchings
    prob = 0
    cvar = 0
    for matching in distinct_matchings:
        stats = get_matching_distr(G, G_nx, matching)
        #print(np.exp(stats[0]+ (stats[1]/2)))
        #print(threshold)
        #print(prob_fast_enough(*stats, threshold))
        curr_prob = prob_fast_enough(*stats, threshold)
        prob += curr_prob
        cvar += fw_right_cvar(*stats, curr_prob)
    
    avg_prob = prob/num_matchings
    avg_cvar = cvar/num_matchings

    return (avg_prob, avg_cvar, avg_run_time, num_matchings)

(G, G_nx) = generate_graph(100, 400, 0.05, 0.5, 0.7)


# print(test_weights(G, G_nx, [1, 0.5], 200, 1000, 7500))
# print(test_weights(G, G_nx, [100, 90], 200, 1000, 7500))

# for warmup in [100, 200, 300, 400, 500, 600, 700, 800]:
#     print("Warmup Period:", warmup)
#     print(test_weights(G, G_nx, [1, 0.5], warmup, itercount=2000, threshold=7500))

