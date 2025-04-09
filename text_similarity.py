import sys

import nltk
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import transformers
import gst_calculation
from sentence_transformers import SentenceTransformer, util


def tokenize_and_tag(text, lang='en'):
    if lang == "de":
        nlp = spacy.load("de_core_news_sm", enable=["tok2vec", "tagger"])
    elif lang == "en":
        nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    else:
        print("lang must be either 'de' or 'en'")
        sys.exit()

    doc = nlp(text)
    token_tag_list = [(token.text, token.tag_) for token in doc]
    return token_tag_list


def length_difference(text1, text2):
    if (len(text1) == 0 and len(text2) == 0):
        return 0
    diff = len(text2) - len(text1)
    return diff
    #return diff, diff/max(len(text1), len(text2)), []


def levenshtein_distance(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


def token_levenshtein_distance(text1, text2, lemmatize=False):
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return levenshtein_distance(tokens1, tokens2)


def __equal_till(s1, s2):
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return i
    return min(len(s1), len(s2))


def longest_common_substring(text1, text2):
    lcs = 0
    for i1 in range(len(text1)):
        for i2 in range(len(text2)):
            l = __equal_till(text1[i1:], text2[i2:])
            lcs = max(lcs, l)
    return lcs


def longest_common_tokensubstring(text1, text2, lemmatize=False):
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return longest_common_substring(tokens1, tokens2)


def gst(text1, text2, minmatch=3):
    _, score = gst_calculation.gst.calculate(text1, text2, minimal_match=minmatch)
    return score


def token_gst(text1, text2, minmatch=3, lemmatize=False):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return gst(tokens1, tokens2, minmatch)


def vector_cosine(text1, text2, lemmatize=False):
    print("text1: ",text1)
    print("text2: ", text2)
    if text1==" " or text2==" ":
        return -1
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens1 = [token for token in tokens1 if token not in stop_words]
    tokens2 = [token for token in tokens2 if token not in stop_words]

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vector1 = vectorizer.fit_transform(tokens1)
    vector2 = vectorizer.transform(tokens2)

    #print(tokens1)
    #print(len(tokens1))
    #print(vector1)
    #print(vector1.shape)
    #print(type(vector1))
    vector1 = np.asarray(vector1.sum(axis=0)[0])
    #print(vector1)
    #print(tokens2)
    #print(vector2)
    #print(vector2.shape)
    vector2 = np.asarray(vector2.sum(axis=0)[0])
    #print(vector2)
    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    return similarity


def sbert_cosine(text1, text2):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    if not isinstance(text1, str) or not isinstance(text2, str):
        return float("nan")
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    cos_sim = util.cos_sim(embedding1, embedding2)
    return cos_sim.item()

if __name__ == '__main__':
    print("Test cases")
    text1 = "Dies ist ein schönes Beispiel. Und noch eins."
    text2 = "Dies ist kein einfaches Beispiel. abc. 123."
    print("Length difference: ", length_difference(text1, text2))
    print("Levenshtein: ", levenshtein_distance(text1, text2))
    print("Levenshtein (Tokens): ", token_levenshtein_distance(text1, text2))
    print("LCS: ", longest_common_substring(text1, text2))
    print("LCS (tokens): ", longest_common_tokensubstring(text1, text2))
    print("GST: ", gst(text1, text2))
    print("GST (Tokens): ", token_gst(text1, text2))
    #print("tf idf - vector cosine: ", vector_cosine(text1, text2))
    print("SBERT - vector cosine: ", sbert_cosine(text1, text2))

    print("finished")