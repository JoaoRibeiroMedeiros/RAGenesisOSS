"""

This Script entails all necessary steps to load the Milvus VectorDB with 
vectorized chunks coming from the texts which make up the knowledge base.

"""

# %%

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np
from src.embedder import create_embeddings_and_save, load_embeddings
from src.chunker import Chunker
import json
import os
from src.utils import get_parameter, setup_path

setup_path()

# %%

load_embeddings_boolean = True
local = True

if local == True:
    host = "localhost"
else:
    hf_api_endpoint = get_parameter("hf_api_endpoint")
    hf_api_key = get_parameter("hf_api_key")
    jina_api_key = get_parameter("jina_api_key")
    host = get_parameter("ragenesis_public_ip")

    os.environ["JINA_API_KEY"] = jina_api_key
    os.environ["HF_API_KEY"] = hf_api_key
    os.environ["HF_API_ENDPOINT"] = hf_api_endpoint

# Documents corpus (replace these with your actual documents)
texts = ["Bible_NT", "Quran", "Torah", "Gita", "Analects"]  # "Bible",

chunker = Chunker()

references_dict, verses_dict = chunker.chunk_all()

all_verses = (
    verses_dict["Bible_NT"]
    + verses_dict["Quran"]
    + verses_dict["Torah"]
    + verses_dict["Gita"]
    + verses_dict["Analects"]
)

verse_token_length_df = chunker.study_verses_length()
# Extracting just the text portions and references from each tuple

verse_token_length_df.describe().to_csv("texts_token_length_describe.csv")

# %%

lengths = [len(references_dict[text]) for text in texts]
all_length = sum(lengths)

print(lengths)
print(all_length)

# %%

# Generate embeddings

encoder_models = ["all_MiniLM_L6_v2", "jina_clip_v1"]
encoder_embedding_dict = {}

for encoder_model in encoder_models:
    embeddings_dict = {}
    if load_embeddings_boolean:
        for text in texts:
            embeddings_dict[text] = load_embeddings(text, encoder_model=encoder_model)
    elif encoder_model == "jina_clip_v1" or encoder_model == "all_MiniLM_L6_v2":
        for text in texts:
            embeddings_dict[text] = create_embeddings_and_save(
                verses_dict[text], text, encoder_model=encoder_model
            )

    encoder_embedding_dict[encoder_model] = embeddings_dict

# %%

print(len(encoder_embedding_dict["jina_clip_v1"]["Bible_NT"]))
print(len(encoder_embedding_dict["jina_clip_v1"]["Bible_NT"][0]))
print(len(encoder_embedding_dict["all_MiniLM_L6_v2"]["Bible_NT"]))
print(len(encoder_embedding_dict["all_MiniLM_L6_v2"]["Bible_NT"][0]))

# %%

# Connect to Milvus
connections.connect(alias="default", host=host, port="19530")

# %%

# Define the schema for your embeddings collection

from_encoder_to_fields = {}
from_encoder_to_schema = {}
from_encoder_to_collection = {}

for encoder_model in encoder_models:

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="holytext", dtype=DataType.VARCHAR, max_length=40),
        FieldSchema(name="reference", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(
            name="verse",
            dtype=DataType.VARCHAR,
            max_length=max([len(verse) for verse in all_verses]) + 5,
        ),
        FieldSchema(name="encoder_model", dtype=DataType.VARCHAR, max_length=40),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=len(encoder_embedding_dict[encoder_model]["Bible_NT"][0]),
        ),
    ]

    from_encoder_to_fields[encoder_model] = fields

    # Create a schema
    schema = CollectionSchema(fields, description="Collection for embeddings")

    from_encoder_to_schema[encoder_model] = schema

    # Create a collection
    collection_name = encoder_model + "_embeddings"

    # Drop the existing collection if it exists
    if utility.has_collection(collection_name):
        Collection(name=collection_name).drop()

    # Create the collection with the new schema
    collection = Collection(name=collection_name, schema=schema)
    from_encoder_to_collection[encoder_model] = collection

    for text in texts:
        partition_name = encoder_model + "_" + text
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)

# %%


def make_insert_data(
    texts,
    references_dict,
    verses_dict,
    encoder_embedding_dict,
    encoder_models=["all_MiniLM_L6_v2", "jina_clip_v1"],
):

    from_encoder_to_insert_data = {}

    for encoder_model in encoder_models:
        insert_data = {}
        for text in texts:
            insert_data[text] = [
                len(references_dict[text]) * [text],
                references_dict[text],
                verses_dict[text],
                len(references_dict[text]) * [encoder_model],
                [x for x in encoder_embedding_dict[encoder_model][text]],
            ]
        from_encoder_to_insert_data[encoder_model] = insert_data

    return from_encoder_to_insert_data


# make data

from_encoder_to_insert_data = make_insert_data(
    texts, references_dict, verses_dict, encoder_embedding_dict, encoder_models
)

# %%

# insert data
for encoder_model in encoder_models:
    print(f"Inserting data for {encoder_model}")
    for text in texts:
        from_encoder_to_collection[encoder_model].insert(
            from_encoder_to_insert_data[encoder_model][text],
            partition_name=encoder_model + "_" + text,
        )
        print("Inserted " + text)

# %%

for encoder_model in encoder_models:
    # Create an IVF_FLAT index for collection.
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1536},
    }

    from_encoder_to_collection[encoder_model].create_index(
        field_name="embedding", index_params=index_params
    )
    from_encoder_to_collection[encoder_model].load()

# %%

for encoder_model in encoder_models:
    # Check insertion
    print(
        f"Number of entities in Milvus: {from_encoder_to_collection[encoder_model].num_entities}"
    )
    print(from_encoder_to_collection[encoder_model].partitions)

# %%

### health check

# Connect to Milvus
connections.connect("default", host=host, port="19530")

# Check if the connection is successful
if connections.has_connection("default"):
    print("Milvus is healthy!")
else:
    print("Milvus is not healthy.")

# %%
