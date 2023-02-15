"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from graphein.protein.analysis import (
    plot_degree_by_residue_type,
    plot_edge_type_distribution,
    plot_residue_composition,
)
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import (
    add_aromatic_interactions,
    add_aromatic_sulphur_interactions,
    add_cation_pi_interactions,
    add_distance_threshold,
    add_disulfide_interactions,
    add_hydrogen_bond_interactions,
    add_ionic_interactions,
    add_k_nn_edges,
    add_peptide_bonds,
)
from graphein.protein.features.nodes.amino_acid import (
    amino_acid_one_hot,
    expasy_protein_scale,
    meiler_embedding,
)
from graphein.protein.graphs import construct_graph
from graphein.protein.utils import download_alphafold_structure
from graphein.protein.visualisation import plot_protein_structure_graph

# Configuration object for graph construction
config = ProteinGraphConfig(
    **{
        "node_metadata_functions": [
            amino_acid_one_hot,
            expasy_protein_scale,
            meiler_embedding,
        ],
        "edge_construction_functions": [
            add_peptide_bonds,
            add_aromatic_interactions,
            add_hydrogen_bond_interactions,
            add_disulfide_interactions,
            add_ionic_interactions,
            add_aromatic_sulphur_interactions,
            add_cation_pi_interactions,
            partial(
                add_distance_threshold, long_interaction_threshold=5, threshold=10.0
            ),
            partial(add_k_nn_edges, k=3, long_interaction_threshold=2),
        ],
    }
)

PDB_CODE = "Q5VSL9"


############## Task 8

##################
protein_path = download_alphafold_structure(
    PDB_CODE, out_dir="/tmp/", aligned_score=False
)
G = construct_graph(pdb_path=protein_path, config=config)
##################

# Print number of nodes and number of edges
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


############## Task 9

##################
degree_sequence = [G.degree(node) for node in G.nodes()]
print(
    "Mean degree: {:.2f}\n".format(np.mean(degree_sequence))
    + "Median degree: {:}\n".format(np.median(degree_sequence))
    + "Max degree: {:}\n".format(np.max(degree_sequence))
    + "Min degree: {:}\n".format(np.min(degree_sequence))
)

fig = plot_degree_by_residue_type(G)
fig.write_image("residue_type.png")
fig = plot_edge_type_distribution(G)
fig.write_image("distribution_type.png")
fig = plot_residue_composition(G)
fig.write_image("composition.png")
plot_protein_structure_graph(G)
##################
