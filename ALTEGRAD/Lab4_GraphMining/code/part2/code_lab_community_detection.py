"""
Graph Mining - ALTEGRAD - Nov 2022
"""

from random import randint

import networkx as nx
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans


############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):

    ##################
    L = nx.laplacian_matrix(G).astype("float")
    val, vec = eigs(L, k=2, which="SR")
    val = val.real
    vec = vec.real
    idx = val.argsort()  # Get indices of sorted eigenvalues
    cluster = KMeans(n_clusters=k)
    Y = cluster.fit_predict(vec[:, idx])
    clustering = {node: Y[i] for i, node in enumerate(G.nodes())}
    ##################

    return clustering


############## Task 7

##################
G = nx.read_edgelist("datasets/CA-HepTh.txt", delimiter="\t")
connected_components = list(nx.connected_components(G))
giant_connected_component = nx.subgraph(G, connected_components[0])
k = 50
clustering = spectral_clustering(giant_connected_component, 50)
##################


############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):

    ##################
    mod = 0
    m = nx.number_of_edges(G)
    for nc in np.unique(list(clustering.values())):
        community = nx.subgraph(
            G, [node for node, cluster in clustering.items() if cluster == nc]
        )
        lc = nx.number_of_edges(community)
        dc = np.sum([community.degree(node) for node in community.nodes()])
        mod += lc / m - (dc / (2 * m)) ** 2
    ##################

    return mod


############## Task 9

##################
print(f"Spectral Clustering modularity: {modularity(G, clustering)}")
random_clusters = np.random.randint(low=0, high=k - 1, size=nx.number_of_nodes(G))
random_clustering = {node: random_clusters[i] for i, node in enumerate(G.nodes())}
print(f"Random Clustering modularity: {modularity(G, random_clustering)}")
##################
