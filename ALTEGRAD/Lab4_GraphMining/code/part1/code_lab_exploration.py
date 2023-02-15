"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

############## Task 1

##################
G = nx.read_edgelist("datasets/CA-HepTh.txt", delimiter="\t")
print("Original graph:", G)
##################


############## Task 2

##################
connected_components = list(nx.connected_components(G))
nb_connected_components = len(connected_components)
giant_connected_component = nx.subgraph(G, connected_components[0])
print("Giant graph:", giant_connected_component)
print(
    "\t{:.2f}% of edges are in the giant\n\t{:.2f}% of nodes are in the giant".format(
        100 * nx.number_of_edges(giant_connected_component) / nx.number_of_edges(G),
        100 * nx.number_of_nodes(giant_connected_component) / nx.number_of_nodes(G),
    )
)
##################


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
print(
    "Statitics on degrees:"
    "\n\tMinimum degree: {}"
    "\n\tMaximum degree: {}"
    "\n\tMean degree: {:.2f}".format(
        np.min(degree_sequence), np.max(degree_sequence), np.mean(degree_sequence)
    )
)
##################


############## Task 4

##################
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].bar(*np.unique(degree_sequence, return_counts=True))
axs[0].set_title("Degree histogram")
axs[0].set_xlabel("Degree")
axs[0].set_ylabel("# of Nodes")

axs[1].bar(*np.unique(degree_sequence, return_counts=True), log=True)
axs[1].set_title("Logarithmic Degree histogram")
axs[1].set_xlabel("Degree")
axs[1].set_ylabel("Log # of Nodes")
##################


############## Task 5

##################
print("Transitivity of original graph: {:.2f}".format(nx.transitivity(G)))
##################
