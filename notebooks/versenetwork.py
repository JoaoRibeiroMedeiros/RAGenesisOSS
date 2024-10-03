# %%


import networkx as nx
from src.retriever import *
from src.rust_network import RustNetworkAnalysis, get_verse_rust_network
from src.network_plots import (
    get_target_verses,
    get_target_node_subgraph,
    plot_centrality_based_subgraph,
)
from src.utils import connect_and_load_milvus_collection

texts = ["Bible_NT", "Quran", "Torah", "Gita", "Analects"]  # "Bible",

encoder_model = "jina_clip_v1"

# encoder_model = "all_MiniLM_L6_v2"

# parameter = "05_threshold"
# similarity_threshold = 0.5

parameter = "075_threshold"
similarity_threshold = 0.75

parameter_path_dict = {"0.5" : "05_threshold" , "0.75" : "075_threshold"}

collection = connect_and_load_milvus_collection(encoder_model=encoder_model)

for text in texts:
    
    target_verses = get_target_verses(collection, [text], encoder_model=encoder_model)

    G_rust = get_verse_rust_network(
        collection,
        [text],
        target_verses,
        encoder_model=encoder_model,
        similarity_threshold=similarity_threshold,
    )

    rust_analysis = RustNetworkAnalysis(
        G_rust, encoder_model=encoder_model, parameter=parameter
    )

    rust_analysis.run()
    rust_analysis.run_stats()
    special_nodes = rust_analysis.get_special_nodes()
    (
        degree_centrality_verse,
        betweenness_centrality_verse,
        closeness_centrality_verse,
        eigenvector_centrality_verse,
    ) = rust_analysis.get_special_verses(target_verses)

    rust_analysis.print_results()
    rust_analysis.save_results(text)
    rust_analysis.save_stats_results(text)

    print("degree_centrality_verse  \n")
    print(degree_centrality_verse, "\n")

    print("betweenness_centrality_verse  \n")
    print(betweenness_centrality_verse, "\n")

    print("closeness_centrality_verse  \n")
    print(closeness_centrality_verse, "\n")

    print("eigenvector_centrality_verse  \n")
    print(eigenvector_centrality_verse, "\n")

    closeness_centrality_verse_index = (
        rust_analysis.calculate_highest_closeness_centrality_index()
    )

    closeness_centrality_subgraph, nodes_to_include, centrality, nodes_data = (
        get_target_node_subgraph(G_rust, closeness_centrality_verse_index)
    )
    plot_centrality_based_subgraph(
        text,
        closeness_centrality_subgraph,
        nodes_data,
        nodes_to_include,
        centrality,
        save=True,
        file_path="data/analytics_data/" + encoder_model + "/" + parameter + "/",
    )

# %%
