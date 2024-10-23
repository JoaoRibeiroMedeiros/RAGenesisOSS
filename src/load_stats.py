import os
import json
import pickle


def load_semantic_network_stats(texts, encoder_model="all_MiniLM_L6_v2", threshold=0.5):

    if threshold == 0.5:
        parameter = "05_threshold"
    elif threshold == 0.75:
        parameter = "075_threshold"

    folder_path = "data/analytics_data/" + encoder_model + "/" + parameter

    file_names = [text + "_network_results.pkl" for text in texts]

    semantic_network_stats = {}

    for text, file_name in zip(texts, file_names):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "rb") as file:
            data = pickle.load(file)
            # text = file_name.split("_")[0]  # convention
            semantic_network_stats[text] = data

    return semantic_network_stats
