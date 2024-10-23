import streamlit as st
import networkx as nx
import pandas as pd
import rustworkx as rx
import altair as alt
import matplotlib.pyplot as plt
import pickle

from pymilvus import connections, utility

from src.retriever import *

from src.utils import (
    reorder_lists,
    remove_chinese_characters,
    organize_centrality_type_occurrence,
)


def get_target_verses(collection, texts, encoder_model="all_MiniLM_L6_v2"):

    # Assuming texts is a list like ['Bible', 'Quran', 'Torah', 'Gita', 'Analects']
    # Convert list of texts into a string suitable for the query expression

    partition_names = [encoder_model + "_" + text for text in texts]

    formatted_texts = ", ".join(f"'{text}'" for text in texts)
    expr = f"holytext in [{formatted_texts}]"

    # Now execute the query
    target_verses = collection.query(
        expr=expr,
        partition_names=partition_names,
        output_fields=["id", "holytext", "reference", "verse", "embedding"],
    )

    return target_verses


def print_basic_graph_stats(G_rust):

    print(G_rust.num_nodes())
    print(G_rust.num_edges())

    rust_components = rx.connected_components(G_rust)
    print(len(list(rust_components)))


def get_target_node_subgraph(G_rust, target_node, method="closeness", lim_neighbors=10):
    """This function returns a subgraph of the target node and
    its top lim_neighbors neighbors based on a centrality metric method, default is closeness centrality.

    Args:
        G_rust (_type_): PyGraph object
        target_node (_type_): node_id
        method (str, optional): centrality metric to be used. Defaults to 'closeness'.
        lim_neighbors (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """

    if method == "degree":
        centralit_dict = {
            n: G_rust.degree(n) / (G_rust.num_nodes() - 1)
            for n in range(G_rust.num_nodes())
        }
    elif method == "betweenness":
        centralit_dict = rx.betweenness_centrality(G_rust)
    elif method == "eigenvector":
        centralit_dict = rx.eigenvector_centrality(G_rust, max_iter=1000)
    elif method == "closeness":
        centralit_dict = rx.closeness_centrality(G_rust)

    # Get neighbors of the target node
    neighbors = G_rust.neighbors(target_node)

    # Get centrality scores for the neighbors
    neighbors_centrality = {
        node: centralit_dict[node] for node in [target_node] + list(neighbors)
    }

    # Convert centrality score dictionary to a list of tuples (node, centrality_score)
    centrality_list = list(neighbors_centrality.items())

    # Sort the list by centrality scores in descending order
    centrality_list.sort(key=lambda x: x[1], reverse=True)

    inherited_centrality = centrality_list[: lim_neighbors + 1]

    most_central_nodes = [node for node, _ in inherited_centrality]

    # Collect nodes for the subgraph: target node and its neighbors
    nodes_to_include = most_central_nodes

    # Create a subgraph including only the specified nodes
    subgraph = G_rust.subgraph(nodes_to_include, preserve_attrs=True)

    nodes_data = {}

    for node in nodes_to_include:

        nodes_data[node] = G_rust.get_node_data(node)

    return subgraph, nodes_to_include, centralit_dict, nodes_data


def get_hc_verses_and_sources_from_references(
    target_verses, high_centrality_references
):
    high_centrality_verses = []
    high_centrality_sources = []
    for high_centrality_reference in high_centrality_references:

        high_centrality_verse = target_verses[
            target_verses["reference"] == high_centrality_reference
        ]["verse"].values[0]
        high_centrality_source = target_verses[
            target_verses["reference"] == high_centrality_reference
        ]["source"].values[0]

        high_centrality_verses.append(high_centrality_verse)
        high_centrality_sources.append(high_centrality_source)

    return high_centrality_verses, high_centrality_sources


def map_full_graph_centrality_in_subgraph_space(dict1, dict2):

    value_mapping = dict1
    value_mapping_2 = dict2

    result = {}
    for i in list(value_mapping.keys()):
        for j in list(value_mapping_2.keys()):
            if i == j:
                result[value_mapping_2[j]] = value_mapping[i]

    return result


def get_node_colors_per_centrality(
    closeness_centrality_subgraph, nodes_data, nodes_to_include, centrality
):
    autumn_r = plt.cm.autumn.reversed()

    node_colors = []
    for i in nodes_to_include:
        if centrality[nodes_to_include[0]] - centrality[nodes_to_include[-1]] == 0:
            normalized_value = 0
        else:
            normalized_value = (centrality[i] - centrality[nodes_to_include[-1]]) / (
                centrality[nodes_to_include[0]] - centrality[nodes_to_include[-1]]
            )

        node_colors.append(autumn_r(normalized_value))

    verses_ordered_by_centrality = list(nodes_data.values())
    verses_ordered_by_subgraph = closeness_centrality_subgraph.nodes()

    node_colors_ordered = reorder_lists(
        verses_ordered_by_centrality, verses_ordered_by_subgraph, node_colors
    )

    return node_colors_ordered


def generate_map_graph_subgraph_labelings(
    closeness_centrality_subgraph, nodes_data, centrality
):
    label_mapping = {
        i: closeness_centrality_subgraph.nodes()[i]
        for i in range(len(closeness_centrality_subgraph.nodes()))
    }
    inverted_label_mapping = {v: k for k, v in label_mapping.items()}
    inverted_nodes_data = {v: k for k, v in nodes_data.items()}

    index_mapping = map_full_graph_centrality_in_subgraph_space(
        inverted_nodes_data, inverted_label_mapping
    )
    label_mapping = {
        i: remove_chinese_characters(label_mapping[i])
        for i in list(index_mapping.keys())
    }

    centrality_label_mapping = {
        i: str(round(centrality[index_mapping[i]], 3))
        for i in range(len(closeness_centrality_subgraph.nodes()))
    }

    return label_mapping, centrality_label_mapping


def plot_centrality_based_subgraph(
    text,
    closeness_centrality_subgraph,
    nodes_data,
    nodes_to_include,
    centrality,
    include_labels=True,
    streamlit=False,
    save=True,
    file_path="data/analytics_data/all_MiniLM_L6_v2/05_threshold/",
):

    label_mapping, centrality_label_mapping = generate_map_graph_subgraph_labelings(
        closeness_centrality_subgraph, nodes_data, centrality
    )

    node_colors = get_node_colors_per_centrality(
        closeness_centrality_subgraph, nodes_data, nodes_to_include, centrality
    )

    edges = closeness_centrality_subgraph.edge_list()

    fig = generate_networkx_plt(
        label_mapping, centrality_label_mapping, node_colors, edges, include_labels
    )

    if save:
        fig.savefig(
            file_path + text + "_degree_centrality_subgraph.png",
            dpi=500,
        )
        with open(
            file_path + text + "_degree_centrality_subgraph_data.pkl", "wb"
        ) as file:
            pickle.dump(
                (label_mapping, centrality_label_mapping, node_colors, edges), file
            )

    if streamlit:
        st.pyplot(fig)


def generate_networkx_plt(
    label_mapping, centrality_label_mapping, node_colors, edges, include_labels=True
):

    G = nx.Graph()

    for node in range(len(list(centrality_label_mapping.values()))):
        # print(node)
        G.add_node(node)

    # Extract nodes and edges from the Rustworkx graph
    # # Add nodes to the NetworkX graph

    for edge in edges:
        # print(edge)
        G.add_edge(edge[0], edge[1])

    # Increase the size of the figure
    fig = plt.figure(figsize=(13, 11))

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes with additional styling
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=800,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.5,
    )

    # Draw edges with additional styling, using undirected graph (no arrows)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color="gray")

    # Create modified positions for label placement
    offset_pos = {node: (x, y - 0.10) for node, (x, y) in pos.items()}

    # Create modified positions for label placement
    offset_pos_metrics = {node: (x, y - 0.15) for node, (x, y) in pos.items()}

    # Draw labels with custom font and better alignment
    if include_labels == True:
        nx.draw_networkx_labels(
            G,
            offset_pos,
            labels=label_mapping,
            font_size=11,
            font_family="Helvetica",
            font_weight="bold",
        )

        nx.draw_networkx_labels(
            G,
            offset_pos_metrics,
            labels=centrality_label_mapping,
            font_size=11,
            font_family="Helvetica",
            font_weight="bold",
        )
    # Add a customized title

    # plt.title('Subgraph around highest closeness centrality verse', fontsize=22, fontname = "Helvetica",fontweight='bold')

    # Adjust the margins to prevent labels from being cut off
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Remove axis
    plt.axis("off")

    plt.tight_layout()

    # Show the plot
    plt.show()

    return fig


def write_main_verses_from_dict(text, network_analytics_dict):

    # Highest Centralities
    st.header("Main Verses")

    st.markdown("Maximum Centrality Nodes in Semantic Network")

    # st.markdown("High Centrality Nodes in Semantic Network")

    special_node_labels = [
        "Highest Degree Centrality",
        "Highest Betweenness Centrality",
        "Highest Closeness Centrality",
        "Highest Eigenvector Centrality",
    ]

    special_nodes = {
        k: network_analytics_dict[k]
        for k in special_node_labels
        if k in network_analytics_dict
    }

    from_centrality_type_to_verse_df = retrieve_special_nodes(text, special_nodes)
    from_references_to_centrality_types, from_verses_to_centrality_types = (
        organize_centrality_type_occurrence(from_centrality_type_to_verse_df)
    )

    for reference, verse in zip(
        list(from_references_to_centrality_types.keys()),
        list(from_verses_to_centrality_types.keys()),
    ):

        st.markdown(f"### **{reference}**")  # reference
        st.markdown(verse)  # verse

        for centrality_type in from_references_to_centrality_types[reference]:

            centrality_key = "Highest " + centrality_type + " Centrality"
            st.markdown(
                f"**Max {centrality_type}**: {network_analytics_dict[centrality_key][1]:.3f}"
            )

    main_references = [
        str(network_analytics_dict["Highest Degree Centrality"][0]),
        str(network_analytics_dict["Highest Betweenness Centrality"][0]),
        str(network_analytics_dict["Highest Closeness Centrality"][0]),
        str(network_analytics_dict["Highest Eigenvector Centrality"][0]),
    ]
    main_verses = [
        str(from_centrality_type_to_verse_df["Degree"]["verse"].values[0]),
        str(from_centrality_type_to_verse_df["Betweenness"]["verse"].values[0]),
        str(from_centrality_type_to_verse_df["Closeness"]["verse"].values[0]),
        str(from_centrality_type_to_verse_df["Eigenvector"]["verse"].values[0]),
    ]

    return main_verses, main_references


def plot_main_network_metrics_from_dict(
    text,
    network_analytics_dict,
    local,
    file_path="data/analytics_data/all_MiniLM_L6_v2/05_threshold/",
):

    with open(file_path + text + "_degree_centrality_subgraph_data.pkl", "rb") as file:
        (
            loaded_label_mapping,
            loaded_centrality_label_mapping,
            loaded_node_colors,
            loaded_edges,
        ) = pickle.load(file)

    fig = generate_networkx_plt(
        loaded_label_mapping,
        loaded_centrality_label_mapping,
        loaded_node_colors,
        loaded_edges,
    )

    st.pyplot(fig)

    # Extract relevant metrics
    metrics = {
        "Average Clustering Coefficient": network_analytics_dict[
            "Average Clustering Coefficient"
        ],
        "Network Density": network_analytics_dict["Network Density"],
        "Average Path Length": network_analytics_dict["Average Path Length"],
        "Average Degree Centrality": network_analytics_dict[
            "Average Degree Centrality"
        ],
        "Average Betweenness Centrality": network_analytics_dict[
            "Average Betweenness Centrality"
        ],
        "Average Closeness Centrality": network_analytics_dict[
            "Average Closeness Centrality"
        ],
        "Average Eigenvector Centrality": network_analytics_dict[
            "Average Eigenvector Centrality"
        ],
        "Max Degree Centrality": network_analytics_dict["Highest Degree Centrality"][1],
        "Max Eigenvector Centrality": network_analytics_dict[
            "Highest Eigenvector Centrality"
        ][1],
        "Max Betweenness Centrality": network_analytics_dict[
            "Highest Betweenness Centrality"
        ][1],
        "Max Closeness Centrality": network_analytics_dict[
            "Highest Closeness Centrality"
        ][1],
    }

    # Bar Chart for Component Length Histogram
    st.header("Component Length Histogram")

    st.markdown(
        f""" 
        Checkout the histogram of component sizes in the network. 

        The size of a component is given by number of verses which are connected to each other in that component. 
        
        Notice metrics given below reflect only the largest component in the network.
        """
    )

    component_histogram_data = pd.DataFrame(
        {
            "Component Size": list(
                network_analytics_dict["Component Length Histogram"].keys()
            ),
            "Frequency": list(
                network_analytics_dict["Component Length Histogram"].values()
            ),
        }
    )

    # CSS for centering
    st.markdown(
        """
        <style>
        .centered {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Use HTML to center the dataframe
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    st.dataframe(component_histogram_data.style.highlight_max(axis=0))
    st.markdown("</div>", unsafe_allow_html=True)

    st.header("Semantic Network Metrics")

    # Convert to network_analytics_dictFrame
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

    # Altair Plot
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("Metric", title="Metrics"),  # Move Metrics to y-axis
            x=alt.X(
                "Value", scale=alt.Scale(type="log"), title="Value (log scale)"
            ),  # Move Value to x-axis
            color=alt.Color("Metric", legend=None),
            tooltip=["Metric", "Value"],
        )
        .properties(
            width=800,  # Increase the width for longer label visibility
            height=400,
            title="Metrics on Logarithmic Scale",
        )
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .configure_title(fontSize=16)
    )

    # Display chart
    st.altair_chart(chart, use_container_width=True)


def get_centrality_types_stats_plot_crosstext(parameter, encoder_model, df):

    import seaborn as sns
    from matplotlib import cm

    df_transposed = df.transpose()

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Create a 2x2 grid of subplots

    # List of column names to plot, assuming they are in your DataFrame
    centrality_types = [
        "degree",
        "eigenvector",
        "betweenness",
        "closeness",
    ]  # Adjust these column names as necessary

    for ax, column in zip(axes.flat, centrality_types):

        # num_bars = df_transposed.shape[0]

        colormap = cm.viridis(np.linspace(0, 0.7, 5))

        # Plot selected column with Viridis colors
        bars = df_transposed.loc["average_" + column + "_centralitys"].plot(
            kind="bar",
            yerr=df_transposed.loc["std_" + column + "_centralitys"],
            capsize=4,
            color=colormap,
            edgecolor="black",
            linewidth=0.7,
            ax=ax,  # Specify the current axis
        )

        # Customize each subplot
        ax.set_title(
            f" {column.capitalize()} Centrality ", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Texts", fontsize=11)
        ax.set_ylabel("Average Value", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=9)
        ax.set_yticklabels(ax.get_yticks(), fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_yscale("log")

        # Remove top and right spines
        sns.despine(ax=ax, left=True, bottom=True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    filepath = (
        "../data/analytics_data/"
        + encoder_model
        + "/"
        + parameter
        + "/crosstext/centrality_comparison.png"
    )

    plt.savefig(filepath)
