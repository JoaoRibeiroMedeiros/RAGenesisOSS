# %%
import streamlit as st

from src.retriever import (
    connect_and_query_holy_texts,
    connect_and_query_holy_texts_ecumenical,
    join_retrieved_references,
    join_central_retrieved_references,
    get_target_node_index,
)

from src.network_plots import (
    plot_centrality_based_subgraph,
    get_target_node_subgraph,
    get_hc_verses_and_sources_from_references,
    write_main_verses_from_dict,
    plot_main_network_metrics_from_dict,
)
from src.generation import get_oracle_response
from src.load_stats import load_semantic_network_stats
from src.utils import reorder_list, get_parameter
from src.load_texts import load_text

import json
import pickle
import os
import random


def sidebar_credits():

    st.sidebar.page_link(
        "https://joaoribeiromedeiros.github.io/",
        label="An app by João Ribeiro Medeiros! ",
    )

    st.sidebar.page_link(
        "https://www.lesswrong.com/posts/rLQGD63B2znREaaP7/retrieval-augmented-genesis",
        label="Read the article!",
    )

    st.sidebar.page_link(
        "https://buymeacoffee.com/joaomedeiros", label="buy me a coffee :) "
    )


def set_query(query):
    st.session_state.query = query

def set_selected_texts(selected_texts):
    st.session_state.selected_texts = selected_texts

def set_selected_text(selected_text):
    st.session_state.selected_text = selected_text

def set_encoder_model(encoder_model):
    st.session_state.encoder_model = encoder_model

def set_parameter(parameter):
    st.session_state.parameter = parameter

def set_verse(verse):
    st.session_state.target_node_reference[st.session_state.selected_text] = verse


st.session_state.counter = 0

holy_texts = ["Bible_NT", "Quran", "Torah", "Gita", "Analects"]

local = os.environ.get("LOCAL", "ec2")

#  AWS secrets !

hf_api_key = get_parameter("hf_api_key")
jina_api_key = get_parameter("jina_api_key")
hf_api_endpoint = get_parameter("hf_api_endpoint")


os.environ["JINA_API_KEY"] = jina_api_key
os.environ["HF_API_KEY"] = hf_api_key
os.environ["HF_API_ENDPOINT"] = hf_api_endpoint


# %%

if "target_node_reference" not in st.session_state:
    st.session_state.target_node_reference = {}

if "target_verses" not in st.session_state:
    st.session_state.target_verses = {}

if "g_rust" not in st.session_state:
    st.session_state.g_rust = {}


st.sidebar.image(
    "images/ragenesis4_edit.png", width=130
)  # Specify height as 200 pixels


st.sidebar.title("RAGENESIS")
st.sidebar.title("Retrieval Augmented Genesis")
st.sidebar.title("- Holy Moly AI!")

# Create a sidebar
st.session_state.page = st.sidebar.selectbox(
    "Select page:", ["VerseUniVerse", "SemanticNetwork", "Bibliography"], index=0
)

encoder_models = ["all_MiniLM_L6_v2", "jina_clip_v1"]


if st.session_state.page != "VerseUniVerse":

    if "selected_text" not in st.session_state:

        st.session_state.selected_text = st.sidebar.selectbox(
            "Select Holy Text", holy_texts, index=random.choice(range(len(holy_texts)))
        )

    else:

        st.session_state.selected_text = st.sidebar.selectbox(
            "Select Holy Text",
            holy_texts,
            index=holy_texts.index(st.session_state.selected_text), on_change=set_selected_text, 
                kwargs={"selected_text": st.session_state.selected_text}

        )

        
         

if st.session_state.page == "VerseUniVerse":

    if "selected_texts" not in st.session_state:

        st.session_state.selected_texts = st.sidebar.multiselect(
            "Select Holy Texts", holy_texts, default=holy_texts
        )

    else:

        st.session_state.selected_texts = st.sidebar.multiselect(
            "Select Holy Texts", holy_texts, default=st.session_state.selected_texts, on_change=set_selected_texts, 
                kwargs={"selected_texts": st.session_state.selected_texts}
        )

        # st.rerun()
         

if st.session_state.page != "Bibliography":

    if "encoder_model" not in st.session_state:
        st.session_state.encoder_model = st.sidebar.selectbox(
            "Select Embedding Model:", encoder_models, index=1
        )
    else:
        st.session_state.encoder_model = st.sidebar.selectbox(
            "Select Embedding Model:",
            encoder_models,
            index=encoder_models.index(st.session_state.encoder_model),on_change=set_encoder_model, 
                kwargs={"encoder_model": st.session_state.encoder_model}
        )
         


def display_retrieval(
    results_sources, results_references, results_verses, target_variable="query"
):
    for source, reference, verse in zip(
        results_sources, results_references, results_verses
    ):
        st.session_state.counter += 1
        if target_variable == "query":

            st.button(
                source + " " + reference,
                key=st.session_state.counter,
                help=None,
                on_click=set_query,
                kwargs={"query": verse},
            )
            st.markdown("")
            st.markdown(verse)
            st.markdown("")

        elif target_variable == "verse":

            st.button(
                source + " " + reference,
                key=st.session_state.counter,
                help=None,
                on_click=set_verse,
                kwargs={"verse": reference},
            )
            st.markdown("")
            st.markdown(verse)
            st.markdown("")


def verse_uni_verse():

    st.title("Verse Uni Verse")

    # st.markdown(
    #     "Navigate through the verses of the selected holy texts based on semantic similarity."
    # )

    st.markdown(
        "This is a search and research tool. Describe a subject or an idea you are interested in. Start a conversation! "
    )

    #    st.markdown(
    #        "The selected holy books will be used as the knowledge base for your search!  "
    #    )

    read_method = st.sidebar.radio(
        "Choose your search method:", ("Open", "Ecumenical"), index=1, key="method"
    )

    oracle = st.sidebar.radio("Toggle GenAI", ("On", "Off"), index=0, key="oracle")

    st.sidebar.markdown(
        """
    Talk to me! I will provide you with food for thought based on the retrieved verses.
                        
    **Enjoy your journey!**            
                
    """
    )

    sidebar_credits()

    if "query" in st.session_state:
        st.session_state.query = st.text_input(
            "Enter Query", value=st.session_state.query
        )
    else:
        st.session_state.query = st.text_input("Enter Query", value="God is love.")

    if st.session_state.method == "Open":
        results_sources, results_references, results_verses = (
            connect_and_query_holy_texts(
                st.session_state.selected_texts,
                st.session_state.query,
                top_k=5,
                local=local,
                encoder_model=st.session_state.encoder_model,
            )
        )
    elif st.session_state.method == "Ecumenical":
        results_sources, results_references, results_verses = (
            connect_and_query_holy_texts_ecumenical(
                st.session_state.selected_texts,
                st.session_state.query,
                top_k=1,
                local=local,
                encoder_model=st.session_state.encoder_model,
            )
        )

    display_retrieval(results_sources, results_references, results_verses)

    if st.session_state.oracle == "On":

        st.title("The Oracle Speaks!")

        with st.spinner("Patience grasshopper..."):

            retrieval = join_retrieved_references(results_references, results_verses)
            response = get_oracle_response(
                st.session_state.query + retrieval, local=local, agent="oracle"
            )

            st.markdown(response)


def semantic_network(local=local):

    parameter_choices = ["0.5", "0.75"]

    parameter_path_dict = {"0.5": "05_threshold", "0.75": "075_threshold"}

    if "parameter" not in st.session_state:

        st.session_state.parameter = st.sidebar.selectbox(
            "Select Similarity threshold:", parameter_choices, index=0
        )

    elif "parameter" in st.session_state:

        st.session_state.parameter = st.sidebar.selectbox(
            "Select Similarity threshold:",
            parameter_choices,
            index=parameter_choices.index(st.session_state.parameter),
            on_change=set_parameter, 
                kwargs={"parameter": st.session_state.parameter}
        )

    network_view = st.sidebar.radio(
        "Choose your Semantic Similarity Network View",
        ("Main Verses", "All Verses"),
        index=1,
        key="semanticview",
    )

    oracle = st.sidebar.radio("Toggle GenAI", ("On", "Off"), index=0, key="oracle")

    st.sidebar.markdown(
        """
    Talk to me! I will provide you with food for thought based on the retrieved verses.
                        
    **Enjoy your journey!**            
                
    """
    )

    sidebar_credits()

    st.title(st.session_state.selected_text)

    if network_view == "Main Verses":

        semantic_network_stats = load_semantic_network_stats(
            [st.session_state.selected_text],
            encoder_model=st.session_state.encoder_model,
            threshold=float(st.session_state.parameter),
        )
        # for text in selected_texts:
        main_verses, main_references = write_main_verses_from_dict(
            st.session_state.selected_text,
            semantic_network_stats[st.session_state.selected_text],
        )

        if st.session_state.oracle == "On":

            st.title("The Exegete Interprets")

            with st.spinner("Patience grasshopper..."):

                retrieval = join_central_retrieved_references(
                    main_references, main_verses
                )
                response = get_oracle_response(retrieval, agent="exegete", local=local)
                st.markdown(response)

        st.title(
            "Highest Closeness Centrality SubGraph for the "
            + st.session_state.selected_text
        )

        plot_main_network_metrics_from_dict(
            st.session_state.selected_text,
            semantic_network_stats[st.session_state.selected_text],
            local,
            file_path="data/analytics_data/"
            + st.session_state.encoder_model
            + "/"
            + parameter_path_dict[st.session_state.parameter]
            + "/",
        )

    elif network_view == "All Verses":

        if (
            st.session_state.encoder_model
            + st.session_state.parameter
            + st.session_state.selected_text
            not in st.session_state.g_rust
        ):
            G_rust, target_verses = load_saved_rust_network(
                selected_text=st.session_state.selected_text,
                encoder_model=st.session_state.encoder_model,
                parameter=parameter_path_dict[st.session_state.parameter],
                local=local,
            )
            st.session_state.g_rust[
                st.session_state.encoder_model
                + st.session_state.parameter
                + st.session_state.selected_text
            ] = G_rust
            st.session_state.target_verses[st.session_state.selected_text] = (
                target_verses
            )

        elif (
            st.session_state.encoder_model
            + st.session_state.parameter
            + st.session_state.selected_text
            in st.session_state.g_rust
        ):
            G_rust = st.session_state.g_rust[
                st.session_state.encoder_model
                + st.session_state.parameter
                + st.session_state.selected_text
            ]

        target_semantic_network(
            G_rust,
            st.session_state.target_verses[st.session_state.selected_text],
        )


def load_saved_rust_network(selected_text, encoder_model, parameter, local=local):

    target_verses = load_text(selected_text)

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

    return G_rust, target_verses


def target_semantic_network(G_rust, target_verses):

    target_verses_references = list(target_verses["reference"])

    if st.session_state.selected_text not in st.session_state.target_node_reference:
        st.session_state.target_node_reference[st.session_state.selected_text] = (
            st.selectbox(
                "Select Verse",
                target_verses_references,
                index=0,
            )
        )
    elif st.session_state.selected_text in st.session_state.target_node_reference:
        st.session_state.target_node_reference[st.session_state.selected_text] = (
            st.selectbox(
                "Select Verse",
                target_verses_references,
                index=target_verses_references.index(
                    st.session_state.target_node_reference[
                        st.session_state.selected_text
                    ]
                ),
            )
        )
         

    target_reference = st.session_state.target_node_reference[
        st.session_state.selected_text
    ]

    with st.spinner("Patience grasshopper..."):

        target_node_index = get_target_node_index(G_rust, target_reference)

        closeness_centrality_subgraph, nodes_to_include, centrality, nodes_data = (
            get_target_node_subgraph(G_rust, target_node=target_node_index)
        )

        plot_centrality_based_subgraph(
            "",
            closeness_centrality_subgraph,
            nodes_data,
            nodes_to_include,
            centrality,
            streamlit=True,
            save=False,
        )

        high_centrality_references = list(nodes_data.values())

        high_centrality_references = reorder_list(
            target_reference, high_centrality_references
        )

        high_centrality_verses, high_centrality_sources = (
            get_hc_verses_and_sources_from_references(
                target_verses, high_centrality_references
            )
        )

        display_retrieval(
            high_centrality_sources,
            high_centrality_references,
            high_centrality_verses,
            target_variable="verse",
        )

        if st.session_state.oracle == "On":

            retrieval = join_retrieved_references(
                high_centrality_references, high_centrality_verses
            )
            response = get_oracle_response(
                st.session_state.target_node_reference[st.session_state.selected_text]
                + retrieval,
                local=local,
                agent="scientist",
            )

            st.title("The Scientist Analyzes")

            st.markdown(response)


def bibliography():

    verse_df = load_text(st.session_state.selected_text)

    st.title("Bibliography")

    with open("Bibliography.md", "r", encoding="utf-8") as file:
        markdown_content = file.read()

    # Display the markdown content in the Streamlit app
    st.markdown(markdown_content)

    sidebar_credits()


if st.session_state.page == "VerseUniVerse":
    verse_uni_verse()
elif st.session_state.page == "SemanticNetwork":
    semantic_network()
elif st.session_state.page == "Bibliography":
    bibliography()
else:
    verse_uni_verse()


# %%
