import string
from collections import defaultdict
import re

#########################################
# a bunch of regex patterns for netspeak#
#########################################

url_pattern = re.compile("http|ht|https|www")
hashtag_pattern = re.compile("^#")
at_pattern = re.compile("^@")
emoji_pattern = re.compile("[:|;][-?)|(|D|/|p|P|3]|;.;|;_;|-_-|<3")
# for matching unicode emoji matching: 
# https://gist.github.com/naotokui/ecce71bcc889e1dc42d20fade74b61e2
unicode_emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)
number_pattern = re.compile("^\$\d+.?\d+|^#\d+|one|two|three|four|five|six|seven|eight|nine|ten|thousand|million|billion|trillion|\d+|\d+?M$", re.IGNORECASE)

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

def tokenize_corpus(corpus, n):
    """
    input : corpus as list of sentences
    output : list of lists of tuples (word, pos)
    """
    processed_corpus = []
    for sentence in corpus:
        if n == 2:
            processed_corpus.append(["<START>"] + tokenize_sentence(sentence) + ["<STOP>"])
        if n == 3:
            processed_corpus.append(["<START1>", "<START2>"] + tokenize_sentence(sentence) + ["<STOP>"])
    return processed_corpus

def replace_pattern_words(corpus):
    """
    removes netspeak patterns from tokenized corpus, replaces with special tokens
    """
    cleaned_corpus = []
    for sentence in dev:
        cleaned_sentence = []
        for word in sentence:

            if word == "<START>" or word == "<STOP>":
                cleaned_sentence.append(word)
            else:
                if url_pattern.match(word[0]):
                    cleaned_sentence.append(("<URL>", word[1]))
                elif hashtag_pattern.match(word[0]):
                    cleaned_sentence.append(("<HASHTAG>", word[1]))
                elif at_pattern.match(word[0]):
                    cleaned_sentence.append(("<AT>", word[1]))
                elif emoji_pattern.match(word[0]):
                    cleaned_sentence.append(("<EMOJI>", word[1]))
                elif unicode_emoji_pattern.match(word[0]):
                    cleaned_sentence.append(("<EMOJI>", word[1]))
                elif number_pattern.match(word[0]):
                    cleaned_sentence.append(("<NUMBER>", word[1]))
                else:
                    cleaned_sentence.append(word)
        cleaned_corpus.append(cleaned_sentence)
    return cleaned_corpus

def count_unigrams(corpus):
    unigram_count = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            unigram_count[word] += 1
    return unigram_count

def extract_vocab(unigram_count, min_count = 1):
    """
    input : defaultdict of unigrams : count
    output : defaultdict with unigrams of low frequency converted to <UNK> : count
    """
    vocab = defaultdict(int)
    for word, count in unigram_count.items():
        if count <= min_count:
            vocab[("<UNK>", word[1])] += min_count
        else: 
            vocab[word] = count
    return vocab, len(vocab)

def set_unk(corpus, vocab):
    """
    input: corpus, list of lists of tuples (word, pos)
    output: corpus with words not in vocab converted to UNK.  list of lists of tuples (<UNK>, pos)
    """
    new_corpus = []
    for sentence in corpus:
        new_sentence = []
        for word in sentence:
            if word not in vocab:
                word = ("<UNK>", word[1])
            new_sentence.append(word)
        new_corpus.append(new_sentence)
    return new_corpus

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('','',string.punctuation))

