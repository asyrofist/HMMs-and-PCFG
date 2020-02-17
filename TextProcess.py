import re
import json
import emoji
from collections import defaultdict

# Regex match for netspeak:
url_pattern = re.compile("http|ht|https|www")
hashtag_pattern = re.compile("^#(?!\d)")
at_pattern = re.compile("^@")
emoji_sideface = re.compile("[8|=|:|;|X|x]-?\s?'?\"?~?[)+|(+|/|D|P|p|3|\||\]|v|V]")
kiss_emoji = re.compile(r"^x$|^[x]+$|[xo]+|<//?3|<+3|<3+", re.IGNORECASE)
emoji_fullface = re.compile("^[.|_]^|;.;|;_+;|-_+-|._+.|T_+T|T.T|\>.\>|\<.\<", re.IGNORECASE)
emoji_backward = re.compile("[\\|/|\(+|\)+|d|D|P|p][:|;|=|8]")
numeric_pattern = re.compile(r"half|zero|one|two|three|four|five|six|seven|eight|nine|ten|twenty|fifty|hundred|thousand|million|billion|trillion|\d+|^[#|\+|\-|x]\d+|\d+[%|\$|M|cm|hr|h|min]$|day\d+|top\d+|\+?[\$|€|£]\d+.?", re.IGNORECASE)


def read_file(path):
    with open(path, "r",encoding='utf-8') as f:
        twts = []
        for line in f:
            twt = json.loads(line)
            twts.append(twt)
        return twts

def convert_emojis(token):
    token = list(token)
    n = len(token)
    
    
    first_index = None
    final_index = 0
    for i in range(n):
        if token[i] in emoji.UNICODE_EMOJI:
            if first_index == None:
                first_index = i
            final_index = i
            
    if first_index != None:
        new_token = token[:first_index] + ["<EMOJI>"] + token[final_index+1:]
    else:
        new_token = token
    return "".join(new_token)

def tokenize_corpus(corpus):
    if len(corpus) > 50000:
        corpus = corpus[:150000]
    new_corpus = []

    for sentence in corpus:
        new_sentence = []
        for word in sentence:
            if url_pattern.match(word[0]):
                word = ("<URL>", word[1])
            elif hashtag_pattern.match(word[0]):
                word = ("<HASHTAG>", word[1])    
            elif at_pattern.match(word[0]):
                word = ("<AT>", word[1])
            elif numeric_pattern.match(word[0]):
                word = ("<NUMBER>",word[1])
            elif emoji_sideface.match(word[0]):
                word = ("<EMOJI>", word[1])
            elif kiss_emoji.match(word[0]):
                word = ("<EMOJI>", word[1])
            elif emoji_fullface.match(word[0]):
                word = ("<EMOJI>", word[1])
            elif emoji_backward.match(word[0]):
                word = ("<EMOJI>", word[1])
            else:
                word = (convert_emojis(word[0]), word[1])
                if "<EMOJI>" not in word[0]:
                    word = (word[0].lower(), word[1])
            new_sentence.append(word)
        new_corpus.append(new_sentence)
    return new_corpus

class ProcessedCorpus:

    def __init__(self, train_path, dev_path, test_path, min_count=1):
        self.raw_train_corpus = read_file(train_path)
        self.raw_dev_corpus = read_file(dev_path)
        self.raw_test_corpus = read_file(test_path)
        self.min_count = min_count

        self.tokenized_train_corpus = tokenize_corpus(self.raw_train_corpus)
        self.tokenized_dev_corpus = tokenize_corpus(self.raw_dev_corpus)
        self.tokenized_test_corpus = tokenize_corpus(self.raw_test_corpus)

        self.normalized_tokenized_train_corpus = []

        # this is from training corpus only
        self.word_count = defaultdict(int)
        self.word_vocab = set()
        self.tag_vocab = set()

        self.word_tag_frequencies()
        self.extract_vocab()

        self.word_vocab_size = len(self.word_vocab)
        self.tag_vocab_size = len(self.tag_vocab)

        self.insert_unk()


    def word_tag_frequencies(self):
        for sentence in self.tokenized_train_corpus:
            for (word, tag) in sentence:
                self.tag_vocab.add(tag)
                self.word_count[word] += 1

    def extract_vocab(self):
        for word, count in self.word_count.items():
            if count > self.min_count:
                self.word_vocab.add(word)

    def insert_unk(self):
        for sentence in self.tokenized_train_corpus:
            new_sentence = []
            for (word, tag) in sentence:
                if word not in self.word_vocab:
                    word = "<UNK>"
                new_sentence.append((word, tag))
            self.normalized_tokenized_train_corpus.append(new_sentence)



