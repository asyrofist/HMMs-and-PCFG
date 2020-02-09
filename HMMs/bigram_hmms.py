import sys, os
sys.path.append("~/Desktop/nlp/HMMs-and-PCFG/HMMs")

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
    p = (bigram_count[(u,v)]+k)/(unigram_count[u]+k*V)
    return p

