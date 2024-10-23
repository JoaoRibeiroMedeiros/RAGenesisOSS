# %%

import re
from pymilvus import connections, Collection
import boto3
import os
import sys


def setup_path():
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_directory not in sys.path:
        sys.path.append(parent_directory)


def get_parameter(name):
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    ssm = boto3.client(
        "ssm",
        region_name="us-east-1",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    response = ssm.get_parameter(
        Name=name, WithDecryption=True
    )  # Assume it is a SecureString
    return response["Parameter"]["Value"]


def connect_and_load_milvus_collection(
    public_ip="localhost", encoder_model="all_MiniLM_L6_v2"
):

    if not connections.has_connection("default"):
        connections.connect(alias="default", host=public_ip, port="19530")

    collection_name = encoder_model + "_embeddings"

    collection = Collection(collection_name)
    collection.load()  # Load collection

    return collection


def reorder_lists(list1, list2, list3):
    if set(list1) != set(list2):
        raise ValueError("list1 and list2 must contain the same elements")
    if len(list1) != len(list3):
        raise ValueError("list1 and list3 must have the same length")

    # Create the reordering dictionary
    reorder_dict = {element: list2.index(element) for element in list1}

    # Create a dictionary mapping elements of list1 to elements of list3
    list1_to_list3 = dict(zip(list1, list3))

    # Apply the reordering to list3
    reordered_list3 = [list1_to_list3[element] for element in list2]

    return reordered_list3


def remove_chinese_characters_from_list(strings):
    non_chinese_list = []
    for string in strings:
        # Use a regular expression to remove Chinese characters
        non_chinese = re.sub(r"[\u4e00-\u9fff]", "", string)
        non_chinese_list.append(non_chinese)
    return non_chinese_list


def remove_chinese_characters(string):

    non_chinese = re.sub(r"[\u4e00-\u9fff]", "", string)

    return non_chinese


# %%


def reorder_list(target, lst):
    # Check if the target is in the list
    if target in lst:
        # Remove the target element
        lst.remove(target)
        # Insert the target at the beginning
        lst.insert(0, target)
    return lst


# %%
def organize_centrality_type_occurrence(from_centrality_type_to_verse_df):

    from_verses_to_centrality_types = {}
    from_references_to_centrality_types = {}

    central_verse_order = {
        0: "Degree",
        1: "Betweenness",
        2: "Closeness",
        3: "Eigenvector",
    }

    verse_df_list = list(from_centrality_type_to_verse_df.values())

    verse_list = [item["verse"].values[0] for item in verse_df_list]

    reference_list = [item["reference"].values[0] for item in verse_df_list]

    # Iterate over each element and its index
    for index, reference, verse in zip(
        range(len(verse_list)), reference_list, verse_list
    ):
        if reference in from_references_to_centrality_types:
            from_references_to_centrality_types[reference].append(
                central_verse_order[index]
            )
            from_verses_to_centrality_types[verse].append(central_verse_order[index])
        else:
            from_references_to_centrality_types[reference] = [
                central_verse_order[index]
            ]
            from_verses_to_centrality_types[verse] = [central_verse_order[index]]

    return from_references_to_centrality_types, from_verses_to_centrality_types
