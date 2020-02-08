from bigrams_hmms import generate_bigrams
from collections import defaultdict

def generate_trigrams(sentence):
    trigrams = []
    for i in range(2, len(sentence)):
        trigram = (sentence[i-2], sentence[i-1], sentence[i])
        trigrams.append(trigram)
    return trigrams

def trigram_proba(u,v,w, bigram_count, trigram_count, k = 0, V):
    p = (trigram_count(u,v,w)+k)/(bigram_count(u,v)+V)
    return p
