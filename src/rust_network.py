import rustworkx as rx
import matplotlib.pyplot as plt
from src.retriever import *
from tqdm import tqdm
import json
from itertools import combinations
from collections import Counter
import numpy as np
import pickle


def print_dict_pretty(d):
    print(json.dumps(d, indent=4))


def get_verse_rust_network(
    collection, texts, target_verses, encoder_model, similarity_threshold=0.5
):

    G = rx.PyGraph(multigraph=False)
    analyzed = set()
    node_index_map = {}

    for verse in target_verses:
        node_index_map[verse["reference"]] = G.add_node(verse["reference"])

    for verse in tqdm(target_verses):
        query_verse_reference = verse["reference"]
        query_verse_embedding = verse["embedding"]
        analyzed.add(query_verse_reference)
        results = retrieve_similar(
            collection,
            query_verse_embedding,
            texts,
            encoder_model=encoder_model,
            top_k=10,
        )
        results_as_dicts = from_query_results_to_dicts(results, scores=True)

        for result in results_as_dicts:
            if (
                result["score"] >= similarity_threshold
                and result["reference"] != query_verse_reference
            ):
                G.add_edge(
                    node_index_map[query_verse_reference],
                    node_index_map[result["reference"]],
                    (query_verse_reference, result["reference"]),
                )  # result["score"])
                # G.add_edge(node_index_map[result['reference']], node_index_map[query_verse_reference], ( result['reference'], query_verse_reference)) # result["score"])

    return G


class RustNetworkAnalysis:

    def __init__(self, G, encoder_model="all_MiniLM_L6_v2", parameter="05_threshold"):
        self.G = G
        self.encoder_model = encoder_model
        self.parameter = parameter

    def run(self):
        print("Calculating average Degree")
        self.average_degree = self.calculate_average_degree()
        print("Running average clustering coefficient")
        self.average_clustering_coefficient = (
            self.calculate_average_clustering_coefficient()
        )
        print("Running network density")
        self.network_density = self.calculate_network_density()

        # slowest is this!
        print("Running average path length")
        self.average_path_length = self.calculate_average_path_length()

        print("Running average degree centrality")
        self.average_degree_centrality = self.calculate_average_degree_centrality()
        print("Running average betweenness centrality")
        self.average_betweenness_centrality = (
            self.calculate_average_betweenness_centrality()
        )
        print("Running average closeness centrality")
        self.average_closeness_centrality = (
            self.calculate_average_closeness_centrality()
        )
        print("Running average eigenvector centrality")
        self.average_eigenvector_centrality = (
            self.calculate_average_eigenvector_centrality()
        )
        print("Running get component length histogram")
        self.component_length_histogram = self.get_component_length_histogram()

        print("Calculating highest degree centrality")
        self.highest_degree_centrality = self.calculate_highest_degree_centrality()

        print("Calculating highest betweenness centrality")
        self.highest_betweenness_centrality = (
            self.calculate_highest_betweenness_centrality()
        )
        print("Calculating highest closeness centrality")
        self.highest_closeness_centrality = (
            self.calculate_highest_closeness_centrality()
        )

        print("Calculating highest eigenvector centrality")
        self.highest_eigenvector_centrality = (
            self.calculate_highest_eigenvector_centrality()
        )

        return self

    def run_stats(self):

        print("Calculating degree centrality histogram")
        self.degree_centrality_histogram = self.calculate_degree_centrality_histogram()

        print("Calculating betweenness centrality histogram")
        self.betweenness_centrality_histogram = (
            self.calculate_betweenness_centrality_histogram()
        )
        print("Calculating closeness centrality histogram")
        self.closeness_centrality_histogram = (
            self.calculate_closeness_centrality_histogram()
        )
        print("Calculating eigenvector centrality histogram")
        self.eigenvector_centrality_histogram = (
            self.calculate_eigenvector_centrality_histogram()
        )

        return self

    def save_results(self, experiment_name="name"):
        results = {
            "Average Degree": self.average_degree,
            "Average Clustering Coefficient": self.average_clustering_coefficient,
            "Network Density": self.network_density,
            "Average Path Length": self.average_path_length,
            "Average Degree Centrality": self.average_degree_centrality,
            "Average Betweenness Centrality": self.average_betweenness_centrality,
            "Average Closeness Centrality": self.average_closeness_centrality,
            "Average Eigenvector Centrality": self.average_eigenvector_centrality,
            "Component Length Histogram": self.component_length_histogram,
            "Highest Degree Centrality": self.highest_degree_centrality,
            "Highest Betweenness Centrality": self.highest_betweenness_centrality,
            "Highest Closeness Centrality": self.highest_closeness_centrality,
            "Highest Eigenvector Centrality": self.highest_eigenvector_centrality,
        }

        with open(
            "data/analytics_data/"
            + self.encoder_model
            + "/"
            + self.parameter
            + "/"
            + experiment_name
            + "_network_results.pkl",
            "wb",
        ) as file:
            pickle.dump(results, file)

    def save_stats_results(self, experiment_name="name"):
        results = {
            "Degree Centrality Histogram Bins": list(
                self.degree_centrality_histogram[1]
            ),
            "Betweenness Centrality Histogram Bins": list(
                self.betweenness_centrality_histogram[1]
            ),
            "Closeness Centrality Histogram Bins": list(
                self.closeness_centrality_histogram[1]
            ),
            "Eigenvector Centrality Histogram Bins": list(
                self.eigenvector_centrality_histogram[1]
            ),
            "Degree Centrality Histogram Heights": list(
                self.degree_centrality_histogram[0]
            ),
            "Betweenness Centrality Histogram Heights": list(
                self.betweenness_centrality_histogram[0]
            ),
            "Closeness Centrality Histogram Heights": list(
                self.closeness_centrality_histogram[0]
            ),
            "Eigenvector Centrality Histogram Heights": list(
                self.eigenvector_centrality_histogram[0]
            ),
            "Degree Centrality Std": self.std_degree_centrality,
            "Betweenness Centrality Std": self.std_betweenness_centrality,
            "Closeness Centrality Std": self.std_closeness_centrality,
            "Eigenvector Centrality Std": self.std_eigenvector_centrality,
        }

        with open(
            "data/analytics_data/"
            + self.encoder_model
            + "/"
            + self.parameter
            + "/"
            + experiment_name
            + "_network_stats_results.pkl",
            "wb",
        ) as file:
            pickle.dump(results, file)

    def print_results(self):
        results = {
            "Average Degree": self.average_degree,
            "Average Clustering Coefficient": self.average_clustering_coefficient,
            "Network Density": self.network_density,
            "Average Path Length": self.average_path_length,
            "Average Degree Centrality": self.average_degree_centrality,
            "Average Betweenness Centrality": self.average_betweenness_centrality,
            "Average Closeness Centrality": self.average_closeness_centrality,
            "Average Eigenvector Centrality": self.average_eigenvector_centrality,
            "Component Length Histogram": self.component_length_histogram,
            "Highest Degree Centrality": self.highest_degree_centrality,
            "Highest Betweenness Centrality": self.highest_betweenness_centrality,
            "Highest Closeness Centrality": self.highest_closeness_centrality,
        }
        print_dict_pretty(results)

    def plot_metrics(self):
        metrics = {
            "Average Clustering Coefficient": self.average_clustering_coefficient,
            "Network Density": self.network_density,
            # "Average Path Length": self.average_path_length,
            "Average Betweenness Centrality": self.average_betweenness_centrality,
        }

        # sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
        # plt.xticks(rotation=45)
        # plt.xlabel("Metrics")
        # plt.ylabel("Values")
        # plt.title("Network Metrics. Average Degree: " + str(self.average_degree))
        # plt.show()
        plt.bar(list(metrics.keys()), list(metrics.values()))
        # Rotate x-ticks
        plt.xticks(rotation=45)
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.title("Network Metrics. Average Degree: " + str(self.average_degree))
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def get_special_nodes(self):
        return {
            "Highest Degree Centrality": self.highest_degree_centrality,
            "Highest Betweenness Centrality": self.highest_betweenness_centrality,
            "Highest Closeness Centrality": self.highest_closeness_centrality,
            "Highest Eigenvector Centrality": self.highest_eigenvector_centrality,
        }

    def get_special_verses(self, target_verses):
        special_nodes = self.get_special_nodes()
        degree_centrality_verse = [
            (verse["reference"], verse["verse"])
            for verse in target_verses
            if verse["reference"] == special_nodes["Highest Degree Centrality"][0]
        ][0]
        betweenness_centrality_verse = [
            (verse["reference"], verse["verse"])
            for verse in target_verses
            if verse["reference"] == special_nodes["Highest Betweenness Centrality"][0]
        ][0]
        closeness_centrality_verse = [
            (verse["reference"], verse["verse"])
            for verse in target_verses
            if verse["reference"] == special_nodes["Highest Closeness Centrality"][0]
        ][0]
        eigenvector_centrality_verse = [
            (verse["reference"], verse["verse"])
            for verse in target_verses
            if verse["reference"] == special_nodes["Highest Eigenvector Centrality"][0]
        ][0]
        return (
            degree_centrality_verse,
            betweenness_centrality_verse,
            closeness_centrality_verse,
            eigenvector_centrality_verse,
        )

    def get_component_length_histogram(self):
        # Get connected components
        components = rx.connected_components(self.G)

        # Calculate the size of each component
        component_sizes = [len(component) for component in components]

        # Generate the histogram of component sizes
        histogram = Counter(component_sizes)

        return histogram

    def calculate_average_degree(self):
        degrees = [self.G.degree(n) for n in range(self.G.num_nodes())]
        return sum(degrees) / self.G.num_nodes()

    def calculate_average_clustering_coefficient(self):
        def local_clustering(node):
            neighbors = list(self.G.neighbors(node))
            if len(neighbors) < 2:
                return 0.0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
            actual_triangles = sum(
                1 for u, v in combinations(neighbors, 2) if self.G.has_edge(u, v)
            )
            return (
                actual_triangles / possible_triangles if possible_triangles > 0 else 0.0
            )

        coefficients = [local_clustering(n) for n in range(self.G.num_nodes())]
        return sum(coefficients) / len(coefficients) if coefficients else 0

    def calculate_network_density(self):
        num_edges = self.G.num_edges()
        num_nodes = self.G.num_nodes()
        return 2.0 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

    def calculate_average_path_length(self):
        def default_weight(edge_data):
            return 1.0

        path_lengths = rx.all_pairs_dijkstra_path_lengths(self.G, default_weight)

        total_length = 0
        count = 0

        for source, lengths in path_lengths.items():
            for target, path_length in lengths.items():
                if source != target:  # Ensure to exclude self-loops
                    total_length += path_length
                    count += 1
        return total_length / count if count > 0 else 0

    def calculate_average_degree_centrality(self):
        degrees = [
            self.G.degree(n) / (self.G.num_nodes() - 1)
            for n in range(self.G.num_nodes())
        ]
        return sum(degrees) / len(degrees) if degrees else 0

    def calculate_average_betweenness_centrality(self):
        centrality = rx.betweenness_centrality(self.G)
        return sum(centrality.values()) / len(centrality) if centrality else 0

    def calculate_average_closeness_centrality(self):
        centrality = rx.closeness_centrality(self.G)
        return sum(centrality.values()) / len(centrality) if centrality else 0

    def calculate_average_eigenvector_centrality(self):
        centrality = rx.eigenvector_centrality(self.G, max_iter=1000)
        return sum(centrality.values()) / len(centrality) if centrality else 0

    def calculate_degree_centrality_histogram(self):
        degrees = [
            self.G.degree(n) / (self.G.num_nodes() - 1)
            for n in range(self.G.num_nodes())
        ]
        self.std_degree_centrality = np.std(degrees)
        return np.histogram(degrees, bins=np.logspace(-10, 0, 100))

    def calculate_betweenness_centrality_histogram(self):
        centrality = rx.betweenness_centrality(self.G)
        self.std_betweenness_centrality = np.std(list(centrality.values()))

        return np.histogram(list(centrality.values()), bins=np.logspace(-10, 0, 100))

    def calculate_closeness_centrality_histogram(self):
        centrality = rx.closeness_centrality(self.G)
        self.std_closeness_centrality = np.std(list(centrality.values()))

        return np.histogram(list(centrality.values()), bins=100, range=(0, 1))

    def calculate_eigenvector_centrality_histogram(self):
        centrality = rx.eigenvector_centrality(self.G, max_iter=1000)
        self.std_eigenvector_centrality = np.std(list(centrality.values()))

        return np.histogram(list(centrality.values()), bins=100, range=(0, 1))

    def calculate_highest_degree_centrality(self):
        degrees = [
            self.G.degree(n) / (self.G.num_nodes() - 1)
            for n in range(self.G.num_nodes())
        ]
        max_index = max(range(len(degrees)), key=degrees.__getitem__)
        return self.G.get_node_data(max_index), degrees[max_index]

    def calculate_highest_betweenness_centrality(self):
        centrality = rx.betweenness_centrality(self.G)
        max_index = max(range(len(centrality)), key=centrality.__getitem__)
        return self.G.get_node_data(max_index), centrality[max_index]

    def calculate_highest_closeness_centrality(self):
        centrality = rx.closeness_centrality(self.G)
        max_index = max(range(len(centrality)), key=centrality.__getitem__)
        return self.G.get_node_data(max_index), centrality[max_index]

    def calculate_highest_eigenvector_centrality(self):
        centrality = rx.eigenvector_centrality(self.G, max_iter=1000)
        max_index = max(range(len(centrality)), key=centrality.__getitem__)
        return self.G.get_node_data(max_index), centrality[max_index]

    def calculate_highest_degree_centrality_index(self):
        degrees = [
            self.G.degree(n) / (self.G.num_nodes() - 1)
            for n in range(self.G.num_nodes())
        ]
        max_index = max(range(len(degrees)), key=degrees.__getitem__)
        return max_index

    def calculate_highest_betweenness_centrality_index(self):
        centrality = rx.betweenness_centrality(self.G)
        max_index = max(range(len(centrality)), key=centrality.__getitem__)
        return max_index

    def calculate_highest_closeness_centrality_index(self):
        centrality = rx.closeness_centrality(self.G)
        max_index = max(range(len(centrality)), key=centrality.__getitem__)
        return max_index

    def calculate_highest_eigenvector_centrality_index(self):
        centrality = rx.eigenvector_centrality_centrality(self.G, max_iter=1000)
        max_index = max(range(len(centrality)), key=centrality.__getitem__)
        return max_index


# Example usage
if __name__ == "__main__":
    G = rx.PyGraph()
    node_a = G.add_node("A")
    node_b = G.add_node("B")
    node_c = G.add_node("C")
    node_d = G.add_node("D")
    G.add_edges_from([(node_a, node_b), (node_a, node_c), (node_c, node_d)])

    analysis = NetworkAnalysis(G)
    analysis.run()

    print("Average Degree:", analysis.average_degree)
    print("Average Clustering Coefficient:", analysis.average_clustering_coefficient)
    print("Density:", analysis.network_density)

    try:
        print("Average Path Length:", analysis.average_path_length)
    except ValueError as e:
        print(e)

    print("Average Betweenness Centrality:", analysis.average_betweenness_centrality)

    node, centrality = analysis.highest_degree_centrality
    print(f"Node with highest degree centrality: {node} (Centrality: {centrality})")

    node, centrality = analysis.highest_betweenness_centrality
    print(
        f"Node with highest betweenness centrality: {node} (Centrality: {centrality})"
    )

    node, centrality = analysis.highest_closeness_centrality
    print(f"Node with highest closeness centrality: {node} (Centrality: {centrality})")
