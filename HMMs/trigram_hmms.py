

def generate_trigrams(sentence):
    trigrams = []
    for i in range(2, len(sentence)):
        trigram = (sentence[i-2], sentence[i-1], sentence[i])
        trigrams.append(trigram)
    return trigrams
