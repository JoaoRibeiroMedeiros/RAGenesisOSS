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
import pickle

texts = ["Bible_NT", "Quran", "Torah", "Gita", "Analects"]  # "Bible",
encoder_models = ["jina_clip_v1", "all_MiniLM_L6_v2"]
parameters = ["05_threshold", "075_threshold"]
parameter_path_dict = {"05_threshold": 0.5, "075_threshold": 0.75}

# %%

for encoder_model in encoder_models:
    collection = connect_and_load_milvus_collection(encoder_model=encoder_model)
    for parameter in parameters:
        similarity_threshold = parameter_path_dict[parameter]
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

            with open(
                "data/analytics_data/"
                + encoder_model
                + "/"
                + parameter
                + "/"
                + text
                + "_graph.pkl",
                "wb",
            ) as f:
                pickle.dump(G_rust, f)


# %%


selected_text = texts[0]
parameter = parameters[0]
encoder_model = encoder_models[0]

with open(
    "data/analytics_data/"
    + encoder_model
    + "/"
    + parameter
    + "/"
    + selected_text
    + "_graph.pkl",
    "rb",
) as f:
    G_rust = pickle.load(f)


# %%

from src.network_plots import get_target_verses
from src.utils import connect_and_load_milvus_collection

selected_text = texts[0]
parameter = parameters[0]
encoder_model = encoder_models[0]

with open("config.json") as f:
    config = json.load(f)
    local = config.get("deploy")
    jina_api_key = config.get("jina_api_key")
    hf_api_key = config.get("hf_api_key")

collection = connect_and_load_milvus_collection(
    public_ip=local, encoder_model=encoder_model
)
target_verses = get_target_verses(
    collection, [selected_text], encoder_model=encoder_model
)
# %%
