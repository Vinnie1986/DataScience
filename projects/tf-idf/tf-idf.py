import math
import os
from collections import defaultdict
from textblob import TextBlob
from projects import ROOT
from pandas import DataFrame


class TfIdfException(Exception):
    pass


class TfIdf:
    """
    short for term frequency–inverse document frequency,
    is a numerical statistic that is intended to reflect how important a word is
    to a document in a collection or corpus.
    """

    def __init__(self):
        self.weighted = False
        self.tf = {}
        self.term_counter = defaultdict(int)
        self.idf = defaultdict(int)

    def get_doc_names(self):
        return self.tf.keys()

    def calculate(self, doc_name):
        """
        calculate the tf-idf of a document
        :param list_of_words:
        :return:
        """
        doc_tf = self.tf.get(doc_name)
        if doc_tf is not None:
            self.calc_idf()
            tf_idf = defaultdict(list)
            for word in doc_tf.keys():
                word_idf = self.idf.get(word)
                word_tf = doc_tf[word]
                tf_idf['word'].append(word)
                tf_idf['tf_idf'].append(word_tf * word_idf)
            return tf_idf
        raise TfIdfException('unknown document number. use the method "add_document_to_the_corpus"')

    def calc_idf(self):
        total_documents = len(self.tf)
        for term in self.term_counter:
            for doc, word_frequency in self.tf.items():
                if term in word_frequency:
                    self.idf[term] += 1
        self.idf.update((x, math.log(total_documents / y)) for x, y in self.idf.items())
        return self.idf

    def add_document_to_the_corpus(self, doc_name, text):
        """
        :param doc_name: the name of a document / paragraph. we use the number.
        :param text: a list of the words in the document / paragraph
        :return:
        """
        words = TextBlob(text).words
        # building a dictionary
        doc_dict = defaultdict(int)
        for word in words:
            doc_dict[word] += 1
            self.term_counter[word] += 1
        # normalizing the dictionary
        length = float(len(words))
        for k in doc_dict:
            doc_dict[k] = doc_dict[k] / length

        # add the normalized document to the corpus
        self.tf[doc_name] = doc_dict


if __name__ == '__main__':
    # read the file
    corpus_path = os.path.join(ROOT, 'tf-idf', 'alice_munro_voices.txt')
    with open(corpus_path, 'r') as corpus_file:
        corpus = corpus_file.read()
        documents = corpus.splitlines()

    # add paragraph per paragraph
    tfidf = TfIdf()
    for i, document in enumerate(documents):
        tfidf.add_document_to_the_corpus(i, document)

    tfidf_doc = tfidf.calculate(doc_name=0)

    # create a dataframe
    df = DataFrame(tfidf_doc)
    top_words = df.sort_values('tf_idf')[-10:]
    # print the words with the most info
    print('the words with the most information of the first paragraph are: \n {}'.format(top_words))
