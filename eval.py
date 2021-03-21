import rouge
import nltk

from summarizers import freq_summary, sempca_summary, transformer_summary
from utils import Results

import matplotlib.pyplot as plt

from tensorflow_datasets.summarization import CnnDailymail

N_DOCS = 10
N_SENTS = 5
metrics = ['rouge-n', 'rouge-l', 'rouge-w']

scorer = rouge.Rouge(metrics=metrics,
                        max_n=3,
                        limit_length=True,
                        length_limit_type='words',
                        apply_avg=False,
                        apply_best=False,
                        alpha=0.5, # Default F1_score
                        weight_factor=1.2,
                        stemming=True)

cnndailymail = CnnDailymail()

texts = []
highlights = []
for doc in cnndailymail.as_dataset()['test']:
    # Transformers library does not accept sequences longer than 1024 tokens
    # so I will skip those longer than 900 words according to NLTK, to account
    # for differences in the tokenization process
    if len(nltk.word_tokenize(doc['article'].numpy().decode())) > 900:
        continue
    texts.append(doc['article'].numpy().decode())
    highlights.append(doc['highlights'].numpy().decode())
    if len(texts) == N_DOCS:
        break

del cnndailymail, doc

results = Results()
for i in range(N_DOCS):
    print("Starting Doc: " + str(i))
    print("Frequency Summary")
    freq_sum = freq_summary(texts[i], N_SENTS)
    print("PCA Summary")
    pca_sums = sempca_summary(texts[i], N_SENTS)
    print("Transformers Summary")
    trans_sum = transformer_summary(texts[i])

    results.update_results(scorer.get_scores([freq_sum], [highlights[i]]), 'freq')
    for j in range(len(pca_sums)):
        results.update_results(scorer.get_scores([pca_sums[j]], [highlights[i]]), 'pca h' + str(j))
    results.update_results(scorer.get_scores([trans_sum], [highlights[i]]), 'transformer')
    print("Finished Doc: " + str(i))

results.plot_results()
results.to_json("results.json")


