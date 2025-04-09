import sys

import pandas as pd
import text_similarity as ts
import data_manager as dm
from tqdm import tqdm
tqdm.pandas()

# TODO in Streamlit: nur die ausgewählten Maße berechnen

all_measures = ["length_diff_tokens",
             "length_diff_chars",
             "levenshtein_tokens",
             "levenshtein_chars",
             "levenshtein_pos",
             "lcs_tokens",
             "lcs_chars",
             "lcs_pos",
             "gst_tokens",
             "gst_chars",
             "gst_pos",
            "sbert_similarity"
            ]

def apply_measures(df, lang, column_text1, column_text2, abs_or_norm, normalize_by, measures, outfilename="results"):

    print("Start")
    df_user = df.copy(deep=True)
    #df = df.head(10).copy()

    t1_tuples = [ts.tokenize_and_tag(text1, lang=lang) for text1 in tqdm(list(df[column_text1]))]
    t2_tuples = [ts.tokenize_and_tag(text2, lang=lang) for text2 in tqdm(list(df[column_text2]))]

    t1_tokens = []
    t2_tokens = []

    t1_pos = []
    t2_pos = []

    for text in tqdm(t1_tuples):

        tokens = [t[0] for t in text]
        pos = [t[1] for t in text]

        t1_tokens.append(tokens)
        t1_pos.append(pos)

    for text in tqdm(t2_tuples):
        tokens = [t[0] for t in text]
        pos = [t[1] for t in text]

        t2_tokens.append(tokens)
        t2_pos.append(pos)

    df["t1_tokens"] = t1_tokens
    df["t2_tokens"] = t2_tokens

    df["t1_pos"] = t1_pos
    df["t2_pos"] = t2_pos

    df["t1_len_chars"] = df[column_text1].apply(len)
    df["t2_len_chars"] = df[column_text2].apply(len)

    df["t1_len_tokens"] = df["t1_tokens"].apply(len)
    df["t2_len_tokens"] = df["t2_tokens"].apply(len)

    if "length_diff_chars" in measures:
        print("length_diff")
        df["length_diff_chars"] = df.progress_apply(lambda x: ts.length_difference(x[column_text1], x[column_text2]), axis=1 )
    if "length_diff_tokens" in measures:
        print("length_diff_tokens")
        df["length_diff_tokens"] = df.progress_apply(lambda x: ts.length_difference(x["t1_tokens"], x["t2_tokens"]), axis=1 )

    if "levenshtein_chars" in measures:
        print("leveshtein_chars")
        df["levenshtein_chars"] = df.progress_apply(lambda x: ts.levenshtein_distance(x[column_text1], x[column_text2]), axis=1 )
    if "levenshtein_tokens" in measures:
        print("leveshtein_tokens")
        df["levenshtein_tokens"] = df.progress_apply(lambda x: ts.levenshtein_distance(x["t1_tokens"], x["t2_tokens"]), axis=1 )
    if "levenshtein_pos" in measures:
        print("leveshtein_pos")
        df["levenshtein_pos"] = df.progress_apply(lambda x: ts.levenshtein_distance(x["t1_pos"], x["t2_pos"]), axis=1 )

    if "lcs_chars" in measures:
        print("lcs_chars")
        df["lcs_chars"] =  df.progress_apply(lambda x: ts.longest_common_substring(x[column_text1], x[column_text2]), axis=1 )
    if "lcs_tokens" in measures:
        print("lcs_tokens")
        df["lcs_tokens"] = df.progress_apply(lambda x: ts.longest_common_substring(x["t1_tokens"], x["t2_tokens"]), axis=1 )
    if "lcs_pos" in measures:
        print("lcs_pos")
        df["lcs_pos"] = df.progress_apply(lambda x: ts.longest_common_substring(x["t1_pos"], x["t2_pos"]), axis=1 )

    if "gst_chars" in measures:
        print("gst_chars")
        df["gst_chars"] = df.progress_apply(lambda x: ts.gst(x[column_text1], x[column_text2]), axis=1)
    if "gst_tokens" in measures:
        print("gst_tokens")
        df["gst_tokens"] = df.progress_apply(lambda x: ts.gst(x["t1_tokens"], x["t2_tokens"]), axis=1)
    if "gst_pos" in measures:
        print("gst_pos")
        df["gst_pos"] = df.progress_apply(lambda x: ts.gst(x["t1_pos"], x["t2_pos"]), axis=1)

    if "sbert_similarity" in measures:
        print("bert")
        df["sbert_similarity"] = df.progress_apply(lambda x: ts.sbert_cosine(x[column_text1], x[column_text2]), axis=1)


    #Normalisieren
    if not abs_or_norm == "abs":
        df["longer"] = "t2"
        df.loc[df["t1_len_chars"] > df["t2_len_chars"], 'longer'] = 't1'

        if normalize_by == "text1":
            df["which_text"] = "t1"
        elif normalize_by == "text2":
           df["which_text"] = "t2"
        else:
            df["which_text"] = df["longer"]

        for measure in measures:
            if measure == "sbert_similarity": continue

            if "chars" in measure:
                df["divide_by"] = df["which_text"].astype(str) + "_len_chars"
            else:
                df["divide_by"] = df["which_text"].astype(str) + "_len_tokens"

            df[measure+"_norm"] = df.apply(lambda x: x[measure]/x[x["divide_by"]], axis=1)




    try:
        normalized_measures = [measure+"_norm" for measure in measures if measure != "sbert_similarity"]

        # include sbert anyways if measures is norm
        if abs_or_norm == "norm":
            if "sbert_similarity" in measures:
                normalized_measures.append("sbert_similarity")
    except:
        normalized_measures = []
        print("Could not normalize measures!")

    if abs_or_norm == "abs":
        relevant_measures = measures
    elif abs_or_norm == "norm":
        relevant_measures = normalized_measures
    else:
        relevant_measures = measures + normalized_measures

    for measure in relevant_measures:
        df_user = pd.concat([df_user, df[measure]], axis=1)


    return df_user



if __name__ == "__main__":

    file = "DARIUS_Energie.csv"
    #file = "DARIUS_Verkehr.csv"
    print("start reading data")
    df_user = dm.load_csv("./data/"+file, sep=";")
    df_user = df_user.loc[(df_user["text1"].apply(len) > 0) & (df_user["text2"].apply(len) > 0)]
    print("finished reading data")
    df_user = apply_measures(df_user,
                   lang="de",
                   column_text1="text1",
                   column_text2="text2",
                   abs_or_norm="both",
                   normalize_by="text1",
                   measures=all_measures)
    #df_user.to_csv("./result/DARIUS_Verkehr_measures" + ".tsv", sep=";", index=False)
    #df_user.to_excel("./result/DARIUS_Verkehr_measures" + ".xlsx")
    df_user.to_csv("./result/DARIUS_Energie_measures" + ".tsv", sep=";", index=False)
    df_user.to_excel("./result/DARIUS_Energie_measures" + ".xlsx")


    # file = "2024_05_16_MS_EMails_gerated_fuer_Andrea.xlsx"
    # print("start reading data")
    # df_user = dm.load_csv("./data/"+file, sep=";")
    # df_user = df_user.loc[(df_user["mail_1"].apply(len) > 0) & (df_user["mail_2"].apply(len) > 0)]
    # print("finished reading data")
    # df_user = apply_measures(df_user,
    #                lang="en",
    #                column_text1="mail_1",
    #                column_text2="mail_2",
    #                abs_or_norm="both",
    #                normalize_by="mail_1",
    #                measures=all_measures)
    # df_user.to_csv("./result/FORMAT_measures" + ".tsv", sep=";", index=False)
    # df_user.to_excel("./result/FORMAT_measures" + ".xlsx")