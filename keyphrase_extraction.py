# example from: http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/#candidate-identification

import re
import string
import operator
import numpy as np
from unidecode import unidecode
from nltk import word_tokenize, sent_tokenize
from nltk import pos_tag_sents
from nltk.chunk.regexp import RegexpParser
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from itertools import chain, groupby
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer

punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))
stop_words = set(stopwords.words('english'))

def generate_candidate(texts, method='word', remove_punctuation=False):
    """
    Generate word candidate from given string

    Parameters
    ----------
    texts: str, input text string
    method: str, method to extract candidate words, either 'word' or 'phrase'

    Returns
    -------
    candidates: list, list of candidate words
    """
    words_ = list()
    candidates = list()

    # tokenize texts to list of sentences of words
    sentences = sent_tokenize(texts)
    for sentence in sentences:
        if remove_punctuation:
            sentence = punct_re.sub(' ', sentence) # remove punctuation
        words = word_tokenize(sentence)
        words = list(map(lambda s: s.lower(), words))
        words_.append(words)
    tagged_words = pos_tag_sents(words_) # POS tagging

    if method == 'word':
        tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'NNPS'])
        tagged_words = chain.from_iterable(tagged_words)
        for word, tag in tagged_words:
            if tag in tags and word.lower() not in stop_words:
                candidates.append(word)
    elif method == 'phrase':
        grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
        chunker = RegexpParser(grammar)
        all_tag = chain.from_iterable([tree2conlltags(chunker.parse(tag)) for tag in tagged_words])
        for key, group in groupby(all_tag, lambda tag: tag[2] != 'O'):
            candidate = ' '.join([word for (word, pos, chunk) in group])
            if key is True and candidate not in stop_words:
                candidates.append(candidate)
    else:
        print("Use either 'word' or 'phrase' in method")
    return candidates


def keyphrase_extraction_tfidf(texts, method='phrase', min_df=5, max_df=0.8, num_key=5):
    """
    Use tf-idf weighting to score key phrases in list of given texts

    Parameters
    ----------
    texts: list, list of texts (remove None and empty string)

    Returns
    -------
    key_phrases: list, list of top key phrases that expain the article

    """
    print('generating vocabulary candidate...')
    vocabulary = [generate_candidate(unidecode(text), method=method) for text in texts]
    vocabulary = list(chain(*vocabulary))
    vocabulary = list(np.unique(vocabulary)) # unique vocab
    print('done!')

    max_vocab_len = max(map(lambda s: len(s.split(' ')), vocabulary))
    tfidf_model = TfidfVectorizer(vocabulary=vocabulary, lowercase=True,
                                  ngram_range=(1,max_vocab_len), stop_words=None,
                                  min_df=min_df, max_df=max_df)
    X = tfidf_model.fit_transform(texts)
    vocabulary_sort = [v[0] for v in sorted(tfidf_model.vocabulary_.items(),
                                            key=operator.itemgetter(1))]
    sorted_array = np.fliplr(np.argsort(X.toarray()))

    # return list of top candidate phrase
    key_phrases = list()
    for sorted_array_doc in sorted_array:
        key_phrase = [vocabulary_sort[e] for e in sorted_array_doc[0:num_key]]
        key_phrases.append(key_phrase)

    return key_phrases


def freqeunt_terms_extraction(texts, ngram_range=(1,1), n_terms=None):
    """
    Extract frequent terms using simple TFIDF ranking in given list of texts
    """
    tfidf_model = TfidfVectorizer(lowercase=True,
                                  ngram_range=ngram_range, stop_words=None,
                                  min_df=5, max_df=0.8)
    X = tfidf_model.fit_transform(texts)
    vocabulary_sort = [v[0] for v in sorted(tfidf_model.vocabulary_.items(),
                                            key=operator.itemgetter(1))]
    ranks = np.array(np.argsort(X.sum(axis=0))).ravel()
    frequent_terms = [vocabulary_sort[r] for r in ranks]
    frequent_terms = [f for f in frequent_terms if len(f) > 3]
    return frequent_terms_filter

if __name__ == '__main__':
    import pandas as pd
    texts = list(pd.read_csv('data/example.txt')['abstract'])
    key_phrases = keyphrase_extraction_tfidf(texts)
    print('Few example key phrases candidate\n')
    for _ in range(3):
        num_doc = np.random.randint(0, len(key_phrases)-1)
        print('abstract: %s' % texts[num_doc])
        print(key_phrases[num_doc])
        print('\n')
