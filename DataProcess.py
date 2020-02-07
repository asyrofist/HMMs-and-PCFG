import string

def read_file(path):
    f = open(path, "r")
    return f.read().splitlines()

def tokenize_sentence(sentence):
    """
    input : sentence in sting format
    output : list of tuples, (word, pos)
    """
    #remove the outer [[ ]] 
    sentence = sentence[2:-2]
    tokens = sentence.split("], [")
    processed_tokens = []
    for token in tokens:
        [word, pos]= token.split(", ")
        # remove quotations around each tokens
        processed_tokens.append((word[1:-1] ,pos[1:-1]))
    return processed_tokens

def tokenize_corpus(corpus):
    """
    input : corpus as list of sentences
    output : list of lists of tuples (word, pos)
    """
    processed_corpus = []
    for sentence in corpus:
        processed_corpus.append(tokenize_sentence(sentence))
    return processed_corpus

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('','',string.punctuation))


