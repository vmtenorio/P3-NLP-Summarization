from nltk.corpus import wordnet, stopwords

import stanza

import matplotlib.pyplot as plt
import numpy as np
import json


class Syns:
    def __init__(self):
        self.syns = {}
        self.inv_syns = {}
        self.total_syns = []

    def get_syns(self, word):
        # If already processed and we have a synonim, return it
        if word in self.total_syns:
            return self.inv_syns[word]
        w_syns = []
        # Add to the data structure
        for synset in wordnet.synsets(word):
            for lem in synset.lemmas():
                w_syns.append(lem.name())
                if lem.name() not in self.inv_syns.keys():
                    self.inv_syns[lem.name()] = word
        w_syns_uniq = list(set(w_syns))
        self.syns[word] = w_syns_uniq
        self.total_syns.extend(w_syns_uniq)
        return word


def proc_text(text):
    stopwords_en = stopwords.words('english')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner,pos,lemma')
    doc = nlp(text)
    tokens = []
    sents = []
    syns = {}
    total_syns = []
    syns = Syns()
    in_ne = False

    # Iterate over the Stanza object
    for s in doc.sentences:
        sent = []
        named_ent = []
        for t in s.tokens:
            # If he is in a named entity, set the flag and go for next token
            if t.ner != 'O':
                in_ne = True 
                named_ent.append(t.text)
                continue
            # Ended named entity
            elif t.ner == 'O' and in_ne:
                in_ne = False
                sent.append(" ".join(named_ent))
                named_ent = []

            # If it is a stopword or not alpha numeric, skip
            if t.text.lower() in stopwords_en or not t.text.isalpha():
                continue

            # Get the word out of the synonims
            w = syns.get_syns(t.words[0].lemma)
            sent.append(w)
        if len(named_ent) > 0:
            sent.append(" ".join(named_ent))
        tokens.append(sent)
        sents.append(s.text)
    tokens_flat = [item for s in tokens for item in s]
    return tokens, tokens_flat, sents


class Results:

    """Util class to store and plot the results obtained by the algorithms"""

    def __init__(self):
        self.results = {}

        # Invariant basic metrics calculated by ROUGE, namely
        # F-Score, Precision and Recall
        self.basic_metrics = ['f', 'p', 'r']

        # Utils for plotting
        self.metric_names = ['Valor-F', 'Precisión', 'Exhaustividad']
        self.colors = ['r', 'b', 'g']

    def init_results(self, scores, sum_type):
        self.results[sum_type] = {}
        # Rouge metrics can change depending on settings, so I initialize them
        # dinamically
        self.rouge_metrics = []
        for k in scores.keys():
            self.results[sum_type][k] = {}
            self.rouge_metrics.append(k)
            for met in scores[k][0].keys():
                self.results[sum_type][k][met] = []

    def update_results(self, scores, sum_type):
        if sum_type not in self.results:
            self.init_results(scores, sum_type)
        for i, rouge_metric in enumerate(scores.keys()):
            for j, metric in enumerate(scores[rouge_metric][0].keys()):
                self.results[sum_type][rouge_metric][metric].append(scores[rouge_metric][0][metric][0])

    def plot_results(self):
        f, ax = plt.subplots(len(self.rouge_metrics), 1, sharex=True, figsize=(12,12))
        x_axis = np.arange(len(self.results))

        for i in range(len(self.rouge_metrics)):
            #ax[i].set_title(self.rouge_metrics[i].capitalize())
            ax[i].set_ylabel(self.rouge_metrics[i].capitalize())
            for j, met in enumerate(self.basic_metrics):
                to_plot = [np.mean(self.results[k][self.rouge_metrics[i]][met]) for k in self.results.keys()]
                ax[i].bar(x=x_axis + .3*(j-1), height=to_plot, color=self.colors[j], width=0.3, label=self.metric_names[j])

        plt.sca(ax[0])
        plt.setp(ax, xticks=x_axis, xticklabels=[x.capitalize() for x in self.results.keys()])
        plt.legend()

        plt.savefig("results.png")

    def to_json(self, path):
        with open(path, 'w') as json_f:
            json.dump(self.results, json_f)

    def from_json(self, path):
        with open(path, 'r') as json_f:
            self.results = json.load(json_f)


