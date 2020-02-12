import sys, os
sys.path.append("~/Desktop/nlp/HMMs-and-PCFG/HMMs")
sys.path.append("~/Desktop/nlp/HMMs-and-PCFG")

from DataProcess import read_file, count_unigrams, extract_vocab, tokenize_corpus, set_unk
from collections import defaultdict

def generate_bigrams(sentence):
    bigrams = []
    for i in range(1,len(sentence)):
        bigram = (sentence[i-1],sentence[i])
        bigrams.append(bigram)
    return bigrams

def count_bigrams(corpus):
    bigram_count = defaultdict(int)
    for sentence in corpus:
        bigrams = generate_bigrams(sentence)
        for bigram in bigrams:
            bigram_count[bigram] += 1
    return bigram_count

def bigram_proba(u, v, unigram_count, bigram_count, V, k = 0):
    if (u,v) not in bigram_count:
        if u not in unigram_count:
            p = 1/V
        else:
            p = k/(unigram_count[v]+k*V)
    else:
        p = (bigram_count[(u,v)]+k)/(unigram_count[u]+k*V)
    return p


def process_train_corpus(train_path):
    train_file = read_file(train_path)
    train_raw = tokenize_corpus(train_file,2)
    unigrams = count_unigrams(train_raw)
    vocab, V = extract_vocab(unigrams)
    train_processed = set_unk(train_raw, vocab)
    return vocab, V, train_processed

class bigramHMMs:
    def __init__(self, train_path, dev_path, test_path, k):
        self.k = k

        # extract vocabulary from train files, process training file with UNK
        self.vocab, self.V, self.train_file = process_train_corpus(train_path)
        # process raw dev and test files
        self.dev_file = read_file(dev_path)
        self.test_file = read_file(test_path)
        
        # count unigrams and bigrams
        self.unigram_count = count_unigrams(self.train_file)
        self.bigram_count = count_bigrams(self.train_file)

        # unigram and bigram probabilities
        self.unigram_probabilities = defaultdict(float)
        self.bigram_probabilities = defaultdict(float)

        # unigram and bigram word count
        self.unigram_wc = defaultdict(int)
        self.bigram_wc = defaultdict(int)

        self.make_bigram_proba_dict(self.train_file)

    def make_bigram_proba_dict(self, corpus):
        for sentence in corpus:
            bigrams = generate_bigrams(sentence)
            for bigram in bigrams:
                self.bigram_probabilities[(bigram[0],bigram[1])] = bigram_proba(bigram[0],bigram[1],self.unigram_count, self.bigram_count, self.V, self.k)
    
    # this would be for transition probabilities
    def unigram_word_count(self):
        self.unigram_wc["<START>"] = self.unigram_count["<START>"]
        self.unigram_wc["<STOP>"] = self.unigram_count["<STOP>"]
        for w,c in self.unigram_count.items():
            if w != "<START>" and w != "<STOP>":
                self.unigram_wc[w] += c

    def bigram_word_count(self):
        return None

    # this would be for emission probabilities
    def tag_count(self):
        return None

    def transition_proba(self):
        return None

    def emission_proba(self):
        return None

    def viterbi_trellis(self):
        return None

    def cell_probabilities(self):
        return None


