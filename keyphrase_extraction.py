# example from: http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/#candidate-identification

import re
import string
from nltk import word_tokenize, sent_tokenize
from nltk import pos_tag_sents
from nltk.chunk.regexp import RegexpParser
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from itertools import chain, groupby
from operator import itemgetter

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
        for key, group in groupby(all_tag, lambda (word, pos, chunk): chunk != 'O'):
            candidate = ' '.join([word for (word, pos, chunk) in group])
            if key is True and candidate not in stop_words:
                candidates.append(candidate)
    else:
        print("Use either 'word' or 'phrase' in method")
    return candidates

if __name__ == '__main__':
    with open ("data/example.txt", "r") as f:
        texts = f.read()
    candidates = generate_candidate(texts, method='phrase')
    print(candidates)
