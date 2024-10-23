# %%

import numpy as np
import pandas as pd
from src.chunker import Chunker


# Documents corpus (replace these with your actual documents)


def make_text_data(
    texts,
    references_dict,
    verses_dict,
):
    text_data = {}
    for text in texts:
        text_data[text] = [
            len(references_dict[text]) * [text],
            references_dict[text],
            verses_dict[text],
        ]

    return text_data


def get_text_df(text, text_data):
    df = pd.DataFrame()
    df["source"] = text_data[text][0]
    df["reference"] = text_data[text][1]
    df["verse"] = text_data[text][2]
    return df


def load_text(text, verbose=False):
    texts = ["Bible_NT", "Quran", "Torah", "Gita", "Analects"]  # "Bible",
    chunker = Chunker()
    references_dict, verses_dict = chunker.chunk_all()
    all_verses = (
        # verses_dict["Bible"]
        verses_dict["Bible_NT"]
        + verses_dict["Quran"]
        + verses_dict["Torah"]
        + verses_dict["Gita"]
        + verses_dict["Analects"]
    )
    lengths = [len(references_dict[text]) for text in texts]
    all_length = sum(lengths)

    if verbose:
        print(lengths)
        print(all_length)

    text_data = make_text_data(texts, references_dict, verses_dict)

    text_df = get_text_df(text, text_data)

    return text_df
