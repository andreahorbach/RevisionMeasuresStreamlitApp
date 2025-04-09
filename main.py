import os

import nltk
import pandas as pd

import data_manager as dm
import parallelprozessing
import text_similarity


def __run_alg(name, alg):
    out = []
    for i in range(len(first_texts)):
        text1 = first_texts[i]
        text2 = second_texts[i]

        if text1 == "" or text2 == "":
            out .append(None)
            continue
        out.append(alg(text1, text2))

    out_df = pd.DataFrame({"id": ids, name: out})
    return out_df


if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

    # load data
    #df = dm.load_csv("./data/FORMAT_Text1_Revision.csv", ";")
    df = dm.load_csv("./data/FORMAT01_MailRevision.csv", ";")

#    first_texts = [m1 for m1 in df["Mail_1"]]
#    second_texts = [m2 for m2 in df["Mail_2"]]
    first_texts = [f"{b1} {m1}" for b1, m1 in zip(df['Betreff_1'], df['Mail_1'])]
    second_texts = [f"{b2} {m2}" for b2, m2 in zip(df['Betreff_2'], df['Mail_2'])]

    ids = [id for id in df["id"]]

    scores = {
        "char-lcs": text_similarity.longest_common_substring,
        "token-lcs": text_similarity.longest_common_tokensubstring,
        "char-gst": text_similarity.gst,
        "token-gst": text_similarity.token_gst,
        "char-levenshtein": text_similarity.levenshtein_distance,
        "token-levenshtein": text_similarity.token_levenshtein_distance,
        "vektor_cos": text_similarity.vector_cosine
    }

    tasks = [(name, scores.get(name)) for name in scores.keys()]

    output = parallelprozessing.run_mono(__run_alg, tasks)

    for out_df in output:
        df = df.join(out_df.set_index('id'), on='id')

    try:
        os.mkdir("./result")
    except FileExistsError:
        pass

    df.to_csv(open("./result/MailRevision.tsv", "w+"), "\t", index=False)

