# TF-IDF

## BEFORE USAGE

run the following:
`pip install -r requirements.txt`
`python -m textblob.download_corpora`


## WHAT?

Suppose you want to summarize a document or a paragraph using few keywords.

One technique is to pick the most frequently occurring terms (words with high term frequency or tf).
However, the most frequent word is a less useful metric since some words like 'this', 'a'  occur very frequently across all documents.

Hence, we also want a measure of how unique a word is i.e. how infrequently the word occurs across all documents (inverse document frequency or idf).

Hence, the product of tf x idf (tfidf) of a word gives a product of how frequent this word is in the document multiplied by how unique the word is w.r.t. the entire corpus of documents.

Words in the document with a high tfidf score occur frequently in the document and provide the most information about that specific document.