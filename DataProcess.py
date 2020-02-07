import string

def read_file(path):
    f = open(path, "r")
    return f.read().splitlines()

def tokenize_sentence(sentence):
    #remove the outer [[ ]] 
    sentence = sentence[2:-2]
    tokens = sentence.split("], [")
    return tokens

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('','',string.punctuation))


