"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deepwalk import deepwalk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Loads the web graph
G = nx.read_weighted_edgelist(
    "../data/web_sample.edgelist", delimiter=" ", create_using=nx.Graph()
)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


############## Task 3
# Extracts a set of random walks from the web graph and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20

##################
model = deepwalk(G, n_walks, walk_length, n_dim)
##################

############## Task 4
# Visualizes the representations of the 100 nodes that appear most frequently in the generated walks
def visualize(model, n, dim):

    nodes = model.wv.index_to_key  # your code here

    ##################
    nodes_frequency = [model.wv.get_vecattr(node, "count") for node in nodes]
    embedding = [
        model.wv[node]
        for node in [nodes[idx] for idx in np.argsort(nodes_frequency)[-100:]]
    ]
    DeepWalk_embeddings = np.array(embedding)  # np.empty(shape=(n, dim))
    ##################

    my_pca = PCA(n_components=10)
    my_tsne = TSNE(n_components=2)

    vecs_pca = my_pca.fit_transform(DeepWalk_embeddings)
    vecs_tsne = my_tsne.fit_transform(vecs_pca)

    fig, ax = plt.subplots()
    ax.scatter(vecs_tsne[:, 0], vecs_tsne[:, 1], s=3)
    for x, y, node in zip(vecs_tsne[:, 0], vecs_tsne[:, 1], nodes):
        ax.annotate(node, xy=(x, y), size=8)
    fig.suptitle("t-SNE visualization of node embeddings", fontsize=30)
    fig.set_size_inches(20, 15)
    plt.savefig("embeddings.pdf")
    plt.show()


visualize(model, 100, n_dim)
