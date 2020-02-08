import sys, os
sys.path.append("~/Desktop/nlp/HMMs-and-PCFG/HMMs")

from .bigram_hmms import generate_bigrams
from collections import defaultdict

def generate_trigrams(sentence):
    trigrams = []
    for i in range(2, len(sentence)):
        trigram = (sentence[i-2], sentence[i-1], sentence[i])
        trigrams.append(trigram)
    return trigrams

def trigram_proba(u,v,w, bigram_count, trigram_count, V, k = 0):
    p = (trigram_count(u,v,w)+k)/(bigram_count(u,v)+V)
    return p
