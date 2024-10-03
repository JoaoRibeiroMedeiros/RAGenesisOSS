import networkx as nx
import matplotlib.pyplot as plt
from src.retriever import *
from tqdm import tqdm
import json
from collections import Counter


def print_dict_pretty(d):
    print(json.dumps(d, indent=4))


def get_verse_network(collection, texts, target_verses, similarity_threshold=0.5):

    G = nx.Graph()
    analyzed = set()

    for verse in target_verses:
        G.add_node(verse["reference"])

    for verse in tqdm(target_verses):
        query_verse_reference = verse["reference"]
        query_verse_embedding = verse["embedding"]
        analyzed.add(query_verse_reference)
        results = retrieve_similar(collection, query_verse_embedding, texts, top_k=10)
        results_as_dicts = from_query_results_to_dicts(results, scores=True)
        # Add all verses as nodes to the graph
        for idx, result in enumerate(results_as_dicts):
            if (
                result["score"] >= similarity_threshold
                and result["reference"] != query_verse_reference
            ):
                G.add_edge(result["reference"], query_verse_reference)

    return G


class NetworkAnalysis:

    def __init__(self, G):
        self.G = G

    def run(self):

        # Averages #
        print("Calculating average Degree")
        self.average_degree = self.calculate_average_degree()
        print("Running average clustering coefficient")
        self.average_clustering_coefficient = (
            self.calculate_average_clustering_coefficient()
        )
        print("Running network density")
        self.network_density = self.calculate_network_density()
        print("Running average path length")
        self.average_path_length = self.calculate_average_path_length()
        print("Running average betweenness centrality")
        self.average_betweenness_centrality = (
            self.calculate_average_betweenness_centrality()
        )
        print("Running average closeness centrality")
        self.average_closeness_centrality = (
            self.calculate_average_closeness_centrality()
        )
        print("Running get component length histogram")
        self.component_length_histogram = self.get_component_length_histogram()

        # Special nodes #
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

        return self

    def save_results(self, experiment_name="name"):
        results = {
            "Average Degree": self.average_degree,
            "Average Clustering Coefficient": self.average_clustering_coefficient,
            "Network Density": self.network_density,
            "Average Path Length": self.average_path_length,
            "Average Betweenness Centrality": self.average_betweenness_centrality,
            "Average Closeness Centrality": self.average_closeness_centrality,
            "Component Length Histogram": self.component_length_histogram,
            "Highest Degree Centrality": self.highest_degree_centrality,
            "Highest Betweenness Centrality": self.highest_betweenness_centrality,
            "Highest Closeness Centrality": self.highest_closeness_centrality,
        }

        with open(experiment_name + "_network_results.json", "w") as file:
            json.dump(results, file)

    def print_results(self):
        results = {
            "Average Degree": self.average_degree,
            "Average Clustering Coefficient": self.average_clustering_coefficient,
            "Network Density": self.network_density,
            "Average Path Length": self.average_path_length,
            "Average Betweenness Centrality": self.average_betweenness_centrality,
            "Average Closeness Centrality": self.average_closeness_centrality,
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
        }

    def get_component_length_histogram(self):
        # Get connected components
        components = nx.connected_components(self.G)

        # Calculate the size of each component
        component_sizes = [len(component) for component in components]

        # Generate the histogram of component sizes
        histogram = Counter(component_sizes)

        return histogram

    def get_connectedness(self):
        components = nx.connected_components(self.G)
        return components

    def calculate_average_degree(self):
        """Calculate and return the average degree of the graph."""
        return sum(dict(self.G.degree()).values()) / float(self.G.number_of_nodes())

    def calculate_average_clustering_coefficient(self):
        """Calculate and return the average clustering coefficient of the graph."""
        return nx.average_clustering(self.G)

    def calculate_network_density(self):
        """Calculate and return the density of the graph."""
        return nx.density(self.G)

    def calculate_average_path_length(self):
        """Calculate and return the average path length of the graph."""
        if nx.is_connected(self.G):
            # If the entire graph is connected, return the average shortest path length.
            return nx.average_shortest_path_length(self.G)
        else:
            # Calculate the average path length for each connected component separately.
            components = nx.connected_components(self.G)
            avg_lengths = []

            for component in components:
                # Create a subgraph for the current component
                subgraph = self.G.subgraph(component)
                # Calculate the average shortest path length for this subgraph
                avg_length = nx.average_shortest_path_length(subgraph)
                avg_lengths.append(avg_length)
            return avg_lengths[0]

    def calculate_average_betweenness_centrality(self):
        """Calculate and return the average betweenness centrality of the graph."""
        betweenness_centralities = nx.betweenness_centrality(self.G)
        return sum(betweenness_centralities.values()) / len(betweenness_centralities)

    def calculate_average_closeness_centrality(self):
        """Calculate and return the average betweenness centrality of the graph."""
        closeness_centralities = nx.closeness_centrality_centrality(self.G)
        return sum(closeness_centralities.values()) / len(closeness_centralities)

    # TODO : add average degree centrality?

    def calculate_highest_degree_centrality(self):
        """Return the node with the highest degree centrality."""
        degree_centrality = nx.degree_centrality(self.G)
        max_node = max(degree_centrality, key=degree_centrality.get)
        return max_node, degree_centrality[max_node]

    def calculate_highest_betweenness_centrality(self):
        """Return the node with the highest betweenness centrality."""
        betweenness_centrality = nx.betweenness_centrality(self.G)
        max_node = max(betweenness_centrality, key=betweenness_centrality.get)
        return max_node, betweenness_centrality[max_node]

    def calculate_highest_closeness_centrality(self):
        """Return the node with the highest closeness centrality."""
        closeness_centrality = nx.closeness_centrality(self.G)
        max_node = max(closeness_centrality, key=closeness_centrality.get)
        return max_node, closeness_centrality[max_node]

    def highest_eigenvector_centrality(self):
        """Return the node with the highest eigenvector centrality."""
        eigenvector_centrality = nx.eigenvector_centrality(self.G)
        max_node = max(eigenvector_centrality, key=eigenvector_centrality.get)
        return max_node, eigenvector_centrality[max_node]

    def highest_katz_centrality(self, alpha=0.1, beta=1.0):
        """Return the node with the highest Katz centrality.

        Parameters:
        alpha: float (optional, default=0.1)
            Attenuation factor; must be less than inverse of largest eigenvalue.
        beta: scalar or dictionary (optional, default=1.0)
            Weight attributed to the immediate neighborhood of each node. If a scalar is specified, it is used as the weight for every node. If a dictionary, it is used as a personalized weight for each node.
        """
        katz_centrality = nx.katz_centrality(self.G, alpha=alpha, beta=beta)
        max_node = max(katz_centrality, key=katz_centrality.get)
        return max_node, katz_centrality[max_node]


# Example usage
if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (3, 4), (4, 5)])

    analysis = NetworkAnalysis(G)

    print("Average Degree:", analysis.average_degree())
    print("Average Clustering Coefficient:", analysis.average_clustering_coefficient())
    print("Density:", analysis.network_density())

    try:
        print("Average Path Length:", analysis.average_path_length())
    except ValueError as e:
        print(e)

    print("Average Betweenness Centrality:", analysis.average_betweenness_centrality())

    node, centrality = analysis.highest_degree_centrality()
    print(f"Node with highest degree centrality: {node} (Centrality: {centrality})")

    node, centrality = analysis.highest_betweenness_centrality()
    print(
        f"Node with highest betweenness centrality: {node} (Centrality: {centrality})"
    )

    node, centrality = analysis.highest_closeness_centrality()
    print(f"Node with highest closeness centrality: {node} (Centrality: {centrality})")
