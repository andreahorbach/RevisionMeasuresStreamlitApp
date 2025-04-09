
from typing import List
import csv
import pandas as pd


def apply_to_col(dataset, col, func, id_col="id",):
    data = dataset[[c for c in dataset.columns if c != col]]
    text = [func(t) for t in dataset[col]]
    ids = [id for id in dataset[id_col]]
    text_df = pd.DataFrame({id_col: ids, col: text})

    data = data.join(text_df.set_index(id_col), on=id_col)

    return data


def filter_df(df, filter, val=True):
    for v, v_df in df.groupby(filter):
        if v == val:
            return v_df


def load_csv(input_path: str, sep: str = None):
    if (input_path.endswith(".xlsx")):
        df = pd.read_excel(input_path)
    else:
        if sep is None:
            if input_path.endswith(".tsv"):
                sep = "\t"
            else:
                sep = ","
        df = pd.read_csv(input_path, sep=sep, low_memory=False)
    df = df.fillna(value="")
    return df
