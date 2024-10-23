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
from tqdm import tqdm
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd


def get_semantic_similarity_network(
    texts, encoder_model, similarity_threshold, save=False, verbose=True
):

    parameter_path_dict = {"0.5": "05_threshold", "0.75": "075_threshold"}

    parameter = parameter_path_dict[str(similarity_threshold)]

    collection = connect_and_load_milvus_collection(encoder_model=encoder_model)

    for text in texts:

        target_verses = get_target_verses(
            collection, [text], encoder_model=encoder_model
        )

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

        if save:
            rust_analysis.save_results(text)
            rust_analysis.save_stats_results(text)

        if verbose:
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


def get_similarity_fraction_at_threshold(
    origin_text, target_text, encoder_model, similarity_threshold, collection
):

    origin_verses = load_text(origin_text)
    target_verses = load_text(target_text)

    origin_text_data = get_embeddings_for_target_text(
        origin_text, collection, encoder_model
    )

    similarity_fraction_at_threshold = 0

    for verse in tqdm(origin_text_data):

        search_vector = np.array(verse["embedding"])

        results = search_and_filter_similar_vectors(
            search_vector, collection, target_text, encoder_model, similarity_threshold
        )

        similarity_fraction_at_threshold += len(results)

    similarity_fraction_at_threshold = similarity_fraction_at_threshold / (
        len(origin_verses) * len(target_verses)
    )

    return similarity_fraction_at_threshold


def get_similarity_fraction_at_threshold_dict_for_all_texts_and_encoder_models(
    encoder_models=["all_MiniLM_L6_v2", "jina_clip_v1"],
    similarity_threshold=0.5,
    texts=["Gita", "Analects"],
):

    from_encoder_model_to_similarity_fraction_at_threshold_dict = {}

    for encoder_model in encoder_models:

        from_encoder_model_to_similarity_fraction_at_threshold_dict[encoder_model] = {}

        collection = connect_and_load_milvus_collection(encoder_model=encoder_model)

        for text_i in tqdm(texts):

            from_encoder_model_to_similarity_fraction_at_threshold_dict[encoder_model][
                text_i
            ] = {}

            for text_j in texts:

                print(text_i, text_j)

                from_encoder_model_to_similarity_fraction_at_threshold_dict[
                    encoder_model
                ][text_i][text_j] = get_similarity_fraction_at_threshold(
                    text_i, text_j, encoder_model, similarity_threshold, collection
                )

    return from_encoder_model_to_similarity_fraction_at_threshold_dict


def get_cross_text_consistency(similarity_fraction_at_threshold_dict):

    cross_text_consistency_formula_1 = {}
    cross_text_consistency_formula_2 = {}
    texts = list(similarity_fraction_at_threshold_dict.keys())
    for text_i in tqdm(texts):
        cross_text_consistency_formula_1[text_i] = {}
        cross_text_consistency_formula_2[text_i] = {}
        for text_j in texts:
            # cross_text_consistency_formula_1[text_i][text_j] = similarity_fraction_at_threshold_dict[text_i][text_j]*log(1/similarity_fraction_at_threshold_dict[text_j][text_j]/similarity_fraction_at_threshold_dict[text_i][text_i])
            # cross_text_consistency_formula_2[text_i][text_j] = similarity_fraction_at_threshold_dict[text_i][text_j]*log(1/similarity_fraction_at_threshold_dict[text_i][text_i]/similarity_fraction_at_threshold_dict[text_j][text_j])
            cross_text_consistency_formula_1[text_i][text_j] = 0.5 * sqrt(
                similarity_fraction_at_threshold_dict[text_i][text_j]
                * similarity_fraction_at_threshold_dict[text_j][text_j]
            ) + 0.5 * sqrt(
                similarity_fraction_at_threshold_dict[text_j][text_i]
                * similarity_fraction_at_threshold_dict[text_i][text_i]
            )
            cross_text_consistency_formula_2[text_i][text_j] = 0.5 * sqrt(
                similarity_fraction_at_threshold_dict[text_i][text_j]
                * similarity_fraction_at_threshold_dict[text_i][text_i]
            ) + 0.5 * sqrt(
                similarity_fraction_at_threshold_dict[text_j][text_i]
                * similarity_fraction_at_threshold_dict[text_j][text_j]
            )

    return cross_text_consistency_formula_1, cross_text_consistency_formula_2


def get_from_encoder_to_cross_text_consistency(
    from_encoder_model_to_similarity_fraction_at_threshold_dict,
):

    from_encoder_to_cross_text_consistency_formula_1 = {}
    from_encoder_to_cross_text_consistency_formula_2 = {}

    for encoder_model in list(
        from_encoder_model_to_similarity_fraction_at_threshold_dict.keys()
    ):

        cross_text_consistency_formula_1, cross_text_consistency_formula_2 = (
            get_cross_text_consistency(
                from_encoder_model_to_similarity_fraction_at_threshold_dict[
                    encoder_model
                ]
            )
        )
        from_encoder_to_cross_text_consistency_formula_1[encoder_model] = (
            cross_text_consistency_formula_1
        )
        from_encoder_to_cross_text_consistency_formula_2[encoder_model] = (
            cross_text_consistency_formula_2
        )

    return (
        from_encoder_to_cross_text_consistency_formula_1,
        from_encoder_to_cross_text_consistency_formula_2,
    )


# texts = ["Bible_NT", "Quran", "Torah", "Gita", "Analects"]  # "Bible",
# encoder_models = ["jina_clip_v1","all_MiniLM_L6_v2" ]
# parameters = [0.5, 0.75]
