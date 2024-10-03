# %%

from src.embedder import encode
from src.load_texts import load_text
from pymilvus import connections, Collection
from src.utils import get_parameter
import random
import requests
import json


def retrieve_similar(collection, query_embedding, holy_texts, encoder_model, top_k):
    """_summary_

    Args:
        collection (_type_): _description_
        query_embedding (_type_): _description_
        holy_texts (_type_): _description_
        encoder_model (_type_): _description_
        top_k (_type_): _description_

    Returns:
        _type_: _description_
    """

    search_params = {
        "metric_type": "COSINE",  # Choose the similarity metric
        "params": {},
    }

    partition_names = [encoder_model + "_" + text for text in holy_texts]

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",  # This should match your field name
        param=search_params,
        limit=top_k,
        expr=None,
        output_fields=["holytext", "reference", "verse", "embedding"],
        partition_names=partition_names,
    )

    return results


def from_query_results_to_dicts(results, scores=False):
    """_summary_

    Args:
        results (_type_): _description_
        scores (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    results_as_dicts = []
    for hits in results:
        for hit in hits:
            if scores:
                hit_dict = {
                    "score": hit.distance,
                    "holytext": hit.entity.holytext,
                    "reference": hit.entity.reference,
                    "verse": hit.entity.verse,
                    "embedding": hit.entity.embedding,  # Access the ID of the hit
                }
            else:
                hit_dict = {
                    "holytext": hit.entity.holytext,
                    "reference": hit.entity.reference,
                    "verse": hit.entity.verse,
                    "embedding": hit.entity.embedding,  # Access the ID of the hit
                }
            results_as_dicts.append(hit_dict)
    return results_as_dicts


def query_holy_text(
    ec2_public_ip,
    query,
    text,
    top_k,
    scores=False,
    encoder_model="all_MiniLM_L6_v2",
):
    """_summary_

    Args:
        ec2_public_ip (_type_): _description_
        query (_type_): _description_
        text (_type_): _description_
        top_k (_type_): _description_
        scores (bool, optional): _description_. Defaults to False.
        encoder_model (str, optional): _description_. Defaults to "sagemaker-huggingface".

    Returns:
        _type_: _description_
    """

    # Connect to Milvus
    connections.connect(alias="default", host=ec2_public_ip, port="19530")

    collection_name = encoder_model + "_embeddings"

    collection = Collection(collection_name)

    query_embedding = encode([query], encoder_model=encoder_model)

    collection.load()  # Load collection

    results = retrieve_similar(
        collection, query_embedding[0], [text], encoder_model, top_k
    )
    results_as_dicts = from_query_results_to_dicts(results, scores)

    return results_as_dicts


def query_many_holy_text(
    ec2_public_ip,
    query,
    holy_texts,
    top_k,
    scores=False,
    encoder_model="all_MiniLM_L6_v2",
):
    """_summary_

    Args:
        ec2_public_ip (_type_): _description_
        query (_type_): _description_
        holy_texts (_type_): _description_
        top_k (_type_): _description_
        scores (bool, optional): _description_. Defaults to False.
        encoder_model (str, optional): _description_. Defaults to "sagemaker-huggingface".

    Returns:
        _type_: _description_
    """

    # Connect to Milvus
    connections.connect("default", host=ec2_public_ip, port="19530")

    collection_name = encoder_model + "_embeddings"
    collection = Collection(collection_name)

    query_embedding = encode([query], encoder_model=encoder_model)

    collection.load()  # Load collection

    results = retrieve_similar(
        collection, query_embedding[0], holy_texts, encoder_model, top_k=top_k
    )
    results_as_dicts = from_query_results_to_dicts(results, scores)

    return results_as_dicts


def connect_and_query_holy_texts(
    holy_texts, query, top_k, local="ec2", encoder_model="all_MiniLM_L6_v2"
):  

    if local == "localdocker":
        ec2_public_ip = "host.docker.internal"
        results_as_dicts = query_many_holy_text(
            ec2_public_ip,
            query,
            holy_texts,
            top_k,
            encoder_model=encoder_model,
        )

    elif local == "localhost":
        ec2_public_ip = "localhost"
        results_as_dicts = query_many_holy_text(
            ec2_public_ip,
            query,
            holy_texts,
            top_k,
            encoder_model=encoder_model,
        )
    elif local == "ec2":

        ec2_public_ip = get_parameter('ragenesis_public_ip')
        results_as_dicts = query_many_holy_text(
            ec2_public_ip,
            query,
            holy_texts,
            top_k,
            encoder_model=encoder_model,
        )

    results_sources = [result["holytext"] for result in results_as_dicts]
    results_references = [result["reference"] for result in results_as_dicts]
    results_verses = [result["verse"] for result in results_as_dicts]

    return results_sources, results_references, results_verses


def connect_and_query_holy_texts_ecumenical(
    holy_texts, query, top_k, local="ec2", encoder_model="all_MiniLM_L6_v2"
):

    results_sources = []
    results_references = []
    results_verses = []

    for text in random.sample(holy_texts, len(holy_texts)):
        if local == "localdocker":
            ec2_public_ip = "host.docker.internal"
            results_as_dicts = query_holy_text(
                ec2_public_ip,
                query,
                text,
                top_k,
                encoder_model=encoder_model,
            )
        elif local == "localhost":
            ec2_public_ip = "localhost"
            results_as_dicts = query_holy_text(
                ec2_public_ip,
                query,
                text,
                top_k,
                encoder_model=encoder_model,
            )
        elif local == "ec2":
            ec2_public_ip = get_parameter('ragenesis_public_ip')
            results_as_dicts = query_holy_text(
                ec2_public_ip,
                query,
                text,
                top_k,
                encoder_model=encoder_model,
            )
        results_sources = results_sources + [
            result["holytext"] for result in results_as_dicts
        ]
        results_references = results_references + [
            result["reference"] for result in results_as_dicts
        ]
        results_verses = results_verses + [
            result["verse"] for result in results_as_dicts
        ]

    return results_sources, results_references, results_verses


def join_retrieved_references(results_references, results_verses):
    consolidated_retrieval = ""
    for reference, verse in zip(results_references, results_verses):
        consolidated_retrieval = (
            consolidated_retrieval + reference + "\n" + verse + "\n\n"
        )
    return consolidated_retrieval

def join_central_retrieved_references(results_references, results_verses):
    order = ["Highest Degree Centrality", "Highest Eigenvector Centrality", "Highest Betweenness Centrality", "Highest  Closeness Centrality"]
    consolidated_retrieval = ""
    for centrality, reference, verse in zip(order, results_references, results_verses):
        consolidated_retrieval = (
             consolidated_retrieval + centrality + " is the following: \n" +reference + "\n"  + verse + "\n\n"
        )
    return consolidated_retrieval


def retrieve_special_nodes_query(special_nodes, local, encoder_model="all_MiniLM_L6_v2"):

    if local == "ec2":
        ec2_public_ip = get_parameter('ragenesis_public_ip')
    else:
        ec2_public_ip = "localhost"
        ec2_public_ip = "localdocker"

    alias_name = "default"

    # Check if the connection with the alias exists
    existing_connection = connections.list_connections()

    # If not connected, establish a new connection
    if existing_connection is None:
        connections.connect(alias=alias_name, host=ec2_public_ip, port="19530")

    collection_name = encoder_model + "_embeddings"

    collection = Collection(collection_name)

    collection.load()  # Load collection

    texts = ["Bible_NT", "Quran", "Torah", "Gita", "Analects"]

    partition_names = [encoder_model + "_" + text for text in texts]

    expr = "reference == '" + special_nodes["Highest Degree Centrality"][0] + "'"
    degree_centrality_verse = collection.query(
        expr=expr,
        partition_names=partition_names,
        output_fields=["id", "holytext", "reference", "verse", "embedding"],
    )

    expr = "reference == '" + special_nodes["Highest Betweenness Centrality"][0] + "'"
    betweenness_centrality_verse = collection.query(
        expr=expr,
        partition_names=partition_names,
        output_fields=["id", "holytext", "reference", "verse", "embedding"],
    )

    expr = "reference == '" + special_nodes["Highest Closeness Centrality"][0] + "'"
    closeness_centrality_verse = collection.query(
        expr=expr,
        partition_names=partition_names,
        output_fields=["id", "holytext", "reference", "verse", "embedding"],
    )

    expr = "reference == '" + special_nodes["Highest Eigenvector Centrality"][0] + "'"
    eigenvector_centrality_verse = collection.query(
        expr=expr,
        partition_names=partition_names,
        output_fields=["id", "holytext", "reference", "verse", "embedding"],
    )

    return (
        degree_centrality_verse,
        betweenness_centrality_verse,
        closeness_centrality_verse,
        eigenvector_centrality_verse,
    )

def retrieve_special_nodes(text, special_nodes):

    text_df = load_text(text)
    centrality_types = ['Degree', 'Betweenness', 'Closeness', 'Eigenvector']
    from_centrality_type_to_verse_df = {}

    for centrality_type in centrality_types:
        node_reference = special_nodes["Highest "+centrality_type+" Centrality"][0]
        special_verse_df = text_df[text_df['reference']==node_reference]
        from_centrality_type_to_verse_df[centrality_type] = special_verse_df

    return from_centrality_type_to_verse_df


def get_target_node_index(G_rust, target_reference):
    target_node_index = [
        node_index
        for node_index in G_rust.node_indexes()
        if G_rust.get_node_data(node_index) == target_reference
    ][0]
    return target_node_index

