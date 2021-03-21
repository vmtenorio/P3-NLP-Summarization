# Summarization using Python

This repository contains the necessary code to implement and test several summarization algorithms, namely:
* A simple algorithm that selects those sentences with the most important words of the text, calculating importance according to the frequency of appearance of thw words in the text.
* A simplified implementation of the algorithm SemPCA, defined in the paper: Alcón, O. and Lloret, E. SemPCA-Summarizer: Exploiting Semantic Principal Component Analysis for Automatic Summary Generation. See [link](https://rua.ua.es/dspace/bitstream/10045/86730/1/2018_Alcon_Lloret_CompInform.pdf)
* A pre-trained transformer that uses the BART algorithm

The structure of files is the following:
* The `utils.py` file contains the method `proc_text` to adequate it for the processing and the `Results` class, to calculate and plot the results.
* The `summarizers.py` file contains all the logic to generate the summaries.
* The `eval.py` file contains the code necessary to evaluate using ROUGE and [py-rouge](https://github.com/Diego999/py-rouge).
* The `NLP-P3.ipynb` file contains all the code to generate and see a summary using the three methods provided.

