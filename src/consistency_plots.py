import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


def plot_heatmap_for_similarity_fraction(
    similarity_fraction_at_threshold_dict, encoder_model, _log=False, threshold=0.75
):

    df = pd.DataFrame(similarity_fraction_at_threshold_dict)

    if _log:
        _df = np.log10(df + 1e-10)
    else:
        _df = df

    # Mask the zero values
    mask = _df == 0

    # Plot the heatmap
    fig = plt.figure(figsize=(8, 6))

    sns.heatmap(
        _df,
        annot=True,
        fmt=".2e",
        cmap="YlGnBu",
        cbar_kws={"label": "Values"},
        mask=mask,
        linewidths=0.5,
        linecolor="black",
    )

    # Add labels
    plt.title(f"Similarity Fraction at Threshold {str(threshold)} - " + encoder_model)
    plt.xlabel("Texts")
    plt.ylabel("Texts")

    # Show the plot
    plt.show()

    return fig


def plot_heatmap_for_similarity_fraction_plotly(
    similarity_fraction_at_threshold_dict, encoder_model, _log=False, threshold=0.75
):
    df = pd.DataFrame(similarity_fraction_at_threshold_dict)

    if _log:
        _df = np.log10(df.replace(0, np.nan) + 1e-10)
    else:
        _df = df

    # Create the heatmap
    fig = px.imshow(
        _df,
        labels=dict(color='Values'),
        x=_df.columns,
        y=_df.index,
        color_continuous_scale="YlGnBu",
        zmin=_df.min().min(),
        zmax=_df.max().max()
    )

    # Update layout to make the plot more informative
    fig.update_layout(
        title=f"Similarity Fraction at Threshold {threshold} - {encoder_model}",
        xaxis_title="Texts",
        yaxis_title="Texts"
    )

    # If you want to hide the hover data where the original value is zero (optional)
    fig.for_each_trace(
        lambda trace: trace.update(hovertemplate=
                                   '<b>%{x}</b><br><b>%{y}</b><br>Value: %{z:.2e}<extra></extra>' if _log else 
                                   '<b>%{x}</b><br><b>%{y}</b><br>Value: %{z:.2f}<extra></extra>')
    )
    
    # Show the figure
    fig.show()


def plot_heatmap_for_cross_text_consistency(
    cross_text_consistency_at_threshold, threshold=0.75
):

    _df = pd.DataFrame(cross_text_consistency_at_threshold)

    # Mask the zero values
    mask = _df == 0
    # Plot the heatmap
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        _df,
        annot=True,
        fmt=".2e",
        cmap="YlOrRd",
        cbar_kws={"label": "Values"},
        mask=mask,
        linewidths=0.5,
        linecolor="black",
    )

    # Add labels
    plt.title(f"Cross-Text Consistency at Threshold {str(threshold)}")
    plt.xlabel("Texts")
    plt.ylabel("Texts")

    # Show the plot
    plt.show()

    return fig



def plot_heatmap_for_cross_text_consistency_plotly(
    cross_text_consistency_at_threshold, threshold=0.75
):
    _df = pd.DataFrame(cross_text_consistency_at_threshold)

    # Create the heatmap
    fig = px.imshow(
        _df,
        labels=dict(color='Values'),
        x=_df.columns,
        y=_df.index,
        color_continuous_scale="YlOrRd",
        zmin=_df.min().min(),
        zmax=_df.max().max()
    )

    # Update layout for better visualization
    fig.update_layout(
        title=f"Cross-Text Consistency at Threshold {threshold}",
        xaxis_title="Texts",
        yaxis_title="Texts"
    )

    # Customize hover information format
    fig.for_each_trace(
        lambda trace: trace.update(hovertemplate=
                                   '<b>%{x}</b><br><b>%{y}</b><br>Value: %{z:.2e}<extra></extra>')
    )
    
    # Show the figure
    fig.show()