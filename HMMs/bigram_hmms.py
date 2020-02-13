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
            # k/kV = 1/V
            p = 1/V
        else:
            p = k/(unigram_count[v]+k*V)
    else:
        p = (bigram_count[(u,v)]+k)/(unigram_count[u]+k*V)
    return p

def emission_proba(w, t, unigram_count, tag_count):
    if (w,t) not in unigram_count:
        p = unigram_count[("<UNK>",t)]/tag_count[t]
    else:
        p = unigram_count[(w,t)]/tag_count[t]
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
        self.unigram_tag_probabilities = defaultdict(float)
        self.bigram_tag_probabilities = defaultdict(float)

        # unigram and bigram tag count
        self.unigram_tc = defaultdict(int)
        self.bigram_tc = defaultdict(int)

        self.unigram_tag_count()
        self.bigram_tag_count()

        # unique tags and unique bigram tags
        self.tags = set(self.unigram_tc.keys())
        self.bigram_tags = set()
        self.generate_bigram_tags()

        self.transition_probabilities = defaultdict(float)
        self.transition_proba()

        self.emission_probabilities = defaultdict(float)

    # this would be for transition probabilities
    def unigram_tag_count(self):
        self.unigram_tc["<START>"] = self.unigram_count["<START>"]
        self.unigram_tc["<STOP>"] = self.unigram_count["<STOP>"]
        for w,c in self.unigram_count.items():
            if w != "<START>" and w != "<STOP>":
                self.unigram_tc[w[1]] += c

    def bigram_tag_count(self):
        for (v,w),c in self.bigram_count.items():
            if v == "<START>":
                self.bigram_tc[(v,w[1])] += c
            elif w == "<STOP>":
                self.bigram_tc[(v[1],w)] += c
            else:
                self.bigram_tc[(v[1],w[1])] += c

    def generate_bigram_tags(self):
        for x in self.tags-{"<STOP>"}:
            for y in self.tags-{"<START>"}:
                self.bigram_tags.add((x,y))

    def transition_proba(self):
        for t in self.bigram_tags:
            self.transition_probabilities[(t[0],t[1])] = \
                bigram_proba(t[0],t[1],self.unigram_tc,self.bigram_tc,len(self.unigram_tc),self.k)

    def viterbi_trellis(self):
        return None

    def cell_probabilities(self):
        return None


