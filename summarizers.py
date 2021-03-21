import nltk
import transformers

from utils import proc_text

from sklearn.decomposition import PCA
import numpy as np

def freq_summary(text, n):
    """
    Returns the summary of the text according to the frequency of the in the whole text
    Sentence importance is calculated as the sum of the frequency of its words
    Picks the n first sentences sorted by importance
    """

    words, words_flattened, sents = proc_text(text)
    word_count = nltk.FreqDist(words_flattened)

    max_freq = max(word_count.values())

    sent_scores = {}
    for i, s in enumerate(sents):
        score = 0
        for w in words[i]:
            score += word_count[w]/max_freq
        sent_scores[s] = score

    sent_scores_sorted = {k: v for k,v in sorted(sent_scores.items(), key=lambda x: x[1], reverse=True)}

    # Get only the n sentences with the highest score
    sents_selected = []
    count = 0
    for k in sent_scores_sorted.keys():
        sents_selected.append(k)
        count += 1
        if count == n:
            break

    # Order the summary as in the original text
    summary = ""
    for s in sents:
        if s in sents_selected:
            summary += s + " "
    return summary[:-1]


def sempca_summary(text, n_sents):
    words, words_flattened, sents = proc_text(text)
    n_words = len(set(words_flattened))

    word_mat = np.zeros((len(sents), len(words_flattened)))
    for i in range(len(sents)):
        for w in words[i]:
            word_mat[i,words_flattened.index(w)] += 1

    # Calculating the Covariance Matrix and PCA from it
    cov_mat = np.cov(word_mat.T)
    pca = PCA(n_components=1)
    cov_pca = pca.fit_transform(cov_mat)
    ord_list = [words_flattened[i] for i in np.argsort(cov_pca.reshape(-1))[::-1]]

    # Heuristic 1 - One sentence in which an important term appears
    sents_h1 = []
    for i in range(n_sents):
        concept = ord_list[i]
        for j in range(len(sents)):
            if concept in words[j] and sents[j] not in sents_h1:
                sents_h1.append(sents[j])
                break
    summary_h1 = " ".join(sents_h1)

    # Heuristic 2 - only the first sentence in which an important term appears
    sents_h2 = []
    idx = 0
    while True:
        if idx == n_sents:
            break
        concept = ord_list[idx]
        for j in range(len(sents)):
            if concept in words[j] and sents[j] not in sents_h1:
                sents_h2.append(sents[j])
                break
            elif concept in words[j] and sents[j] in sents_h1:
                break
        idx += 1
    summary_h2 = " ".join(sents_h2)

    # Heuristic 3 - all the sentences in which the most important term appears
    sents_h3 = []
    concept = ord_list[0]
    for j in range(len(sents)):
        if concept in words[j]:
            sents_h3.append(sents[j])
    summary_h3 = " ".join(sents_h3)

    # Heuristic 4 - all the sentences in which the two most important terms appear
    sents_h4 = []
    concept_1 = ord_list[0]
    concept_2 = ord_list[1]
    for j in range(len(sents)):
        if concept_1 in words[j] and concept_2 in words[j]:
            sents_h4.append(sents[j])
    summary_h4 = " ".join(sents_h4)

    return summary_h1, summary_h1, summary_h3, summary_h4

def transformer_summary(text):
    summarizer = transformers.pipeline("summarization")
    summary_trans = summarizer(text)
    return summary_trans[0]['summary_text']
