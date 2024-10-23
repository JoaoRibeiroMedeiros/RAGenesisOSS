import numpy as np
import pandas as pd

from src.retriever import (
    get_embeddings_for_target_text,
    search_and_filter_similar_vectors,
)
from src.utils import connect_and_load_milvus_collection

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

from src.retriever import *
from tqdm import tqdm

from math import log, sqrt

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

from sklearn.decomposition import PCA

origin_text = "Gita"


def pca_embedding_per_text_analysis(collection, text, n_components=3):

    origin_text_data = get_embeddings_for_target_text(text, collection)
    embeddings_df = pd.DataFrame(
        [origin_text_data[i]["embedding"] for i in range(len(origin_text_data))]
    )

    # Create a new DataFrame with the PCA results
    pca = PCA(n_components=n_components)  # Reduce to 2 dimensions for visualization
    pca_result = pca.fit_transform(embeddings_df)

    _columns = ["PCA" + str(i) for i in range(n_components)]

    # Create a new DataFrame with the PCA results
    pca_df = pd.DataFrame(pca_result, columns=_columns)
    explained_variance = pca.explained_variance_ratio_

    return pca_df, explained_variance


def get_all_embeddings(collection, texts, encoder_model="all_MiniLM_L6_v2"):

    all_embeddings_df = pd.DataFrame()

    for text in texts:
        origin_text_data = get_embeddings_for_target_text(
            text, collection, encoder_model=encoder_model
        )
        embeddings_df = pd.DataFrame(
            [origin_text_data[i]["embedding"] for i in range(len(origin_text_data))]
        )
        embeddings_df["text"] = text
        embeddings_df = embeddings_df.T
        all_embeddings_df = pd.concat([all_embeddings_df, embeddings_df], axis=1)

    return all_embeddings_df


def pca_embedding_per_all_texts_analysis(all_embeddings_df, n_components=3):

    text_labels = all_embeddings_df.loc["text"]
    # print('asd')
    _all_embeddings_df = all_embeddings_df.drop(index="text")

    # Create a new DataFrame with the PCA results
    pca = PCA(n_components=n_components)  # Reduce to 2 dimensions for visualization
    pca_result = pca.fit_transform(_all_embeddings_df.T)

    _columns = ["PCA" + str(i) for i in range(n_components)]
    # Create a new DataFrame with the PCA results
    pca_df = pd.DataFrame(pca_result, columns=_columns)

    # Option 1: Reset index if necessary
    pca_df = pca_df.reset_index(drop=True)
    text_labels = text_labels.reset_index(drop=True)

    # Option 2: Ensure alignment
    pca_df["text"] = text_labels.values

    explained_variance = pca.explained_variance_ratio_

    return pca_df, explained_variance


def plot_embedding_pca_results(title, pca_df, explained_variance):
    # Plotting the PCA results

    fig = plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="PCA0", y="PCA1", hue="text", data=pca_df, palette="viridis", s=50, alpha=0.3
    )
    plt.title("PCA of " + title + " Embeddings")
    plt.xlabel(f"PCA0 - {explained_variance[0]*100:.2f}%")
    plt.ylabel(f"PCA1 - {explained_variance[1]*100:.2f}%")
    plt.grid(True)

    # plt.savefig()
    return fig


import plotly.express as px


def plot_interactive_embedding_pca_results(title, pca_df, explained_variance):
    # Create an interactive scatter plot with Plotly
    fig = px.scatter(
        pca_df,
        x="PCA0",
        y="PCA1",
        color="text",
        opacity=0.3,
        title=f"PCA of " + title + " Embeddings",
        labels={
            "PCA0": f"PCA0 - {explained_variance[0] * 100:.2f}%",
            "PCA1": f"PCA1 - {explained_variance[1] * 100:.2f}%",
        },
        width=1000,
        height=700,
    )

    # Update layout to ensure grid is visible
    fig.update_layout(
        xaxis_title=f"PCA0 - {explained_variance[0]*100:.2f}%",
        yaxis_title=f"PCA1 - {explained_variance[1]*100:.2f}%",
        xaxis_showgrid=True,
        yaxis_showgrid=True,
    )

    return fig


def make_pyplot_plots(pca_dfs, explained_variances, encoder_models):

    for encoder_model in encoder_models:
        # To display the figure, you would call:
        fig = plot_interactive_embedding_pca_results(
            encoder_model, pca_dfs[encoder_model], explained_variances[encoder_model]
        )
        fig.show()
        fig.write_html(encoder_model + "_PCA_embeddings_holy_books.html")

    return pca_dfs


def get_pca_for_knowledgebase(
    texts, encoder_models=["all_MiniLM_L6_v2", "jina_clip_v1"], n_components=2
):

    pca_dfs = {}
    explained_variances = {}

    for encoder_model in encoder_models:

        collection = connect_and_load_milvus_collection(encoder_model=encoder_model)
        all_embeddings_df = get_all_embeddings(
            collection, texts, encoder_model=encoder_model
        )
        pca_df, explained_variance = pca_embedding_per_all_texts_analysis(
            all_embeddings_df, n_components=n_components
        )
        pca_dfs[encoder_model], explained_variances[encoder_model] = (
            pca_df,
            explained_variance,
        )

    return all_embeddings_df, pca_dfs, explained_variances


def make_sns_plot(pca_dfs, explained_variances, encoder_model):

    fig = plot_embedding_pca_results(
        encoder_model + "", pca_dfs[encoder_model], explained_variances[encoder_model]
    )
    plt.show()

    return fig


def get_convex_hull_hypervolume(pca_dfs, encoder_model, texts):

    from_text_to_semantic_hypervolume = {}
    for text in texts:
        text_data = pca_dfs[encoder_model][pca_dfs[encoder_model]["text"] == text]
        text_data = text_data.drop(columns="text")
        hull = ConvexHull(text_data)
        volume = hull.volume
        from_text_to_semantic_hypervolume[text] = volume

    return from_text_to_semantic_hypervolume


def get_convex_hull_hypervolume_per_embeddingmodel(pca_dfs, encoder_models, texts):
    from_encoder_model_to_from_text_to_semantic_hypervolume = {}
    for encoder_model in encoder_models:
        from_encoder_model_to_from_text_to_semantic_hypervolume[encoder_model] = (
            get_convex_hull_hypervolume(pca_dfs, encoder_model, texts)
        )
    return from_encoder_model_to_from_text_to_semantic_hypervolume


def normalize_per_token_size(from_encoder_model_to_from_text_to_semantic_hypervolume):
    df_tokens_in_texts = pd.read_csv("texts_token_length_describe.csv").set_index(
        "Unnamed: 0"
    )
    from_encoder_model_to_from_text_to_semantic_hypervolume_normalized_by_text_size = {}
    for encoder_model in list(
        from_encoder_model_to_from_text_to_semantic_hypervolume.keys()
    ):
        from_encoder_model_to_from_text_to_semantic_hypervolume_normalized_by_text_size[
            encoder_model
        ] = {}
        for text in list(
            from_encoder_model_to_from_text_to_semantic_hypervolume[
                encoder_model
            ].keys()
        ):
            from_encoder_model_to_from_text_to_semantic_hypervolume_normalized_by_text_size[
                encoder_model
            ][
                text
            ] = (
                1000000
                * from_encoder_model_to_from_text_to_semantic_hypervolume[
                    encoder_model
                ][text]
                / df_tokens_in_texts.loc["count"][text]
            )
    return (
        from_encoder_model_to_from_text_to_semantic_hypervolume_normalized_by_text_size
    )


def plot_cumulative_explained_variance(explained_variances):
    # Create a plot with enhanced aesthetics
    plt.figure(figsize=(12, 8))

    encoder_models = list(explained_variances.keys())
    # Using a loop to plot data for each encoder model with a gradient color
    colors = cm.get_cmap("winter", len(encoder_models))  # Create a colormap

    for idx, model in enumerate(encoder_models):
        cumsum_variance = np.cumsum(
            explained_variances[model]
        )  # Calculate cumulative sum for each model
        plt.plot(
            cumsum_variance, label=model, color=colors(idx)
        )  # Plot with label and color
    plt.title("Cumulative Sum of Explained Variances for Encoder Models", fontsize=18)
    plt.xlabel("PCA Dimension", fontsize=18)
    plt.ylabel("Cumulative Sum of Explained Variance", fontsize=18)

    # Add grid for better readability
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Add a legend to distinguish between the encoder models
    plt.legend(title="Encoder Models", fontsize=16, title_fontsize="14")

    # Optimize layout
    plt.tight_layout()

    # Display the plot
    plt.show()

def plot_cumulative_explained_variance_plotly(explained_variances):
    # Initialize a figure
    fig = go.Figure()

    encoder_models = list(explained_variances.keys())
    # Define a color palette
    colors = px.colors.sequential.Winter

    # Plot data for each encoder model
    for idx, model in enumerate(encoder_models):
        cumsum_variance = np.cumsum(explained_variances[model])
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumsum_variance) + 1)),
                y=cumsum_variance,
                mode='lines+markers',
                name=model,
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(symbol='circle', size=8)
            )
        )

    # Update the layout for better aesthetics
    fig.update_layout(
        title="Cumulative Sum of Explained Variances for Encoder Models",
        xaxis_title="PCA Dimension",
        yaxis_title="Cumulative Sum of Explained Variance",
        template="plotly_white",
        legend_title_text="Encoder Models",
        hovermode="x unified",
        font=dict(size=14)
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')

    # Show the interactive plot
    fig.show()

# Example usage with your data:
# plot_cumulative_explained_variance(explained_variances)

def plot_semantic_hyper_volume_per_text(
    from_encoder_model_to_from_text_to_semantic_hypervolume,
    title="Semantic Hyper Volume",
):

    data = from_encoder_model_to_from_text_to_semantic_hypervolume
    # Extract categories and values for both models
    categories = list(data["all_MiniLM_L6_v2"].keys())
    values_minilm = list(data["all_MiniLM_L6_v2"].values())
    values_jina_clip = list(data["jina_clip_v1"].values())

    # Set up the bar plot
    x = np.arange(len(categories))  # Label locations
    width = 0.35  # The width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(
        x - width / 2, values_minilm, width, label="all_MiniLM_L6_v2", color="#4C72B0"
    )
    rects2 = ax.bar(
        x + width / 2, values_jina_clip, width, label="jina_clip_v1", color="#55A868"
    )

    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_xlabel("Texts", fontsize=14)
    ax.set_ylabel(title, fontsize=14)
    ax.set_title(title + " in PCA space by Text and Model", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=12)

    # Add a grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Annotate bars with values
    ax.bar_label(rects1, padding=3, fmt="%.3f", fontsize=10)
    ax.bar_label(rects2, padding=3, fmt="%.3f", fontsize=10)

    # Make the plot more compact
    plt.tight_layout()
    plt.show()


import plotly.graph_objects as go


def plot_semantic_hyper_volume_per_text_plotly(data, title="Semantic Hyper Volume"):
    # Extract categories and values for both models
    categories = list(data["all_MiniLM_L6_v2"].keys())
    values_minilm = list(data["all_MiniLM_L6_v2"].values())
    values_jina_clip = list(data["jina_clip_v1"].values())

    # Create bar traces
    trace_minilm = go.Bar(
        x=categories, y=values_minilm, name="all_MiniLM_L6_v2", marker_color="#4C72B0"
    )
    trace_jina_clip = go.Bar(
        x=categories, y=values_jina_clip, name="jina_clip_v1", marker_color="#55A868"
    )

    # Set up the layout
    layout = go.Layout(
        title=f"{title} in PCA space by Text and Model",
        xaxis=dict(title="Texts"),
        yaxis=dict(title=title),
        barmode="group",
        bargap=0.15,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1,  # gap between bars of the same location coordinates
    )

    # Create a figure with the data and layout
    fig = go.Figure(data=[trace_minilm, trace_jina_clip], layout=layout)

    # Show the figure
    fig.show()


# Example of how to use this function with your data
# plot_semantic_hyper_volume_per_text(from_encoder_model_to_from_text_to_semantic_hypervolume)
