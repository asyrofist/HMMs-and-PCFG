
def generate_bigrams(sentence):
    bigrams = []
    for i in range(1,len(sentence)):
        bigram = (sentence[i-1],sentence[i])
        bigrams.append(bigram)
    return bigrams
