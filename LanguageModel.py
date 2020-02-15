from TextProcess import ProcessedCorpus
from collections import defaultdict

def generate_ngram(n, sentence):
    n_grams = []
    for i in range(len(sentence)-n+1):
        n_grams.append(tuple(sentence[i:i+n]))
    return n_grams

def pad_corpus(n, corpus):
    # pad corpus with <START> and <STOP>
    new_corpus = []
    for sentence in corpus:
        new_corpus.append([("<START>","<START>")]*(n-1) + sentence+ [("<STOP>","<STOP>")])
    return new_corpus

#corpus = ProcessedCorpus(train, dev, test, min_count)

class LanguageModel:

    def __init__(self, corpus):
        # unigram, bigram, trigram language model
        # all the corpuses
        self.train = corpus.normalized_tokenized_train_corpus
        self.dev = corpus.tokenized_dev_corpus
        self.test = corpus.tokenized_test_corpus

        self.bigram_padded_train = pad_corpus(2, self.train)
        self.trigram_padded_train = pad_corpus(3, self.train)

        # smoothing parameters
        self.k = 1
        self.l1 = 0
        self.l2 = 0
        self.l3 = 0

        # unigrams
        self.unigram_count = defaultdict(int)
        self.unigram_tc = defaultdict(int)
        self.unigram_wc = defaultdict(int)
        self.count_unigrams()

        # bigrams
        self.bigram_count = defaultdict(int)
        self.bigram_tc = defaultdict(int)
        self.bigram_wc = defaultdict(int)
        self.count_bigrams()

        # trigrams
        self.trigram_count = defaultdict(int)
        self.trigram_tc = defaultdict(int)
        self.trigram_wc = defaultdict(int)
        self.count_trigrams()

        # all possible unigrams, bigrams, trigrams
        self.all_possible_unigram_tags = set(self.unigram_tc.keys())
        self.all_possible_bigram_tags = set()
        self.all_possible_trigram_tags = set()
        self.generate_all_possible_bigrams()
        self.generate_all_possible_trigrams()

        # relevant vocab and sizes
        self.tag_len = len(self.unigram_tc.keys())
        self.total = sum(self.unigram_count.values())
        self.tags = set(self.unigram_tc.keys())

        # mle probabilities
        self.unigram_mles = defaultdict(float)
        self.bigram_mles = defaultdict(float)
        self.trigram_mles = defaultdict(float)
        self.bigram_mle_dict()
        self.trigram_mle_dict()

    def set_smoothing_params(self, k, l1, l2, l3):
        self.k = k
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def count_unigrams(self):
        for sentence in self.bigram_padded_train:
            for (word, tag) in sentence:
                self.unigram_tc[tag] += 1
                self.unigram_wc[word] += 1
                self.unigram_count[(word,tag)] += 1

    def count_bigrams(self):
        for sentence in self.bigram_padded_train:
            bigrams = generate_ngram(2, sentence)
            for (wx,tx),(wy,ty) in bigrams:
                self.bigram_tc[(tx,ty)] += 1
                self.bigram_wc[(wx,wy)] += 1
                self.bigram_count[((wx,tx),(wy,ty))] += 1

    def count_trigrams(self):
        for sentence in self.trigram_padded_train:
            trigrams = generate_ngram(3, sentence)
            for (wx,tx),(wy,ty),(wz,tz) in trigrams:
                self.trigram_tc[(tx,ty,tz)] += 1
                self.trigram_wc[(wx,wy,wz)] += 1
                self.trigram_count[((wx,tx),(wy,ty),(wz,tz))] += 1

    def unigram_mle(self, w, count_dict, V):
        if w not in count_dict:
            p_mle = count_dict["<UNK>"]/V
        else:
            p_mle = count_dict[w]/V
        return p_mle

    def ngram_mle(self, priorgram, ngram, priorgram_dict, ngram_dict, V):
        if ngram not in ngram_dict:
            if priorgram not in priorgram_dict:
                p_mle = 1/V
            else:
                p_mle = self.k/(priorgram_dict[priorgram]+self.k*V)
        else:
            p_mle = (ngram_dict[ngram]+self.k)/(priorgram_dict[priorgram]+self.k*V)
        return p_mle

    def generate_all_possible_bigrams(self):
        for t1 in self.all_possible_unigram_tags:
            for t2 in self.all_possible_unigram_tags:
                self.all_possible_bigram_tags.add((t1,t2))

    def generate_all_possible_trigrams(self):
        for t1 in self.all_possible_unigram_tags:
            for t2 in self.all_possible_unigram_tags:
                for t3 in self.all_possible_unigram_tags:
                    self.all_possible_trigram_tags.add((t1,t2,t3))

    def unigram_mle_dict(self):
        for t in self.tags:
            self.unigram_mles[t] = self.unigram_mle(t, self.unigram_tc, self.total)

    def bigram_mle_dict(self):
        for t in self.all_possible_bigram_tags:
            self.bigram_mles[t] = self.ngram_mle(t[0],t,self.unigram_tc, self.bigram_tc, self.tag_len)

    def trigram_mle_dict(self):
        for t in self.all_possible_trigram_tags:
            self.trigram_mles[t] = self.ngram_mle((t[0],t[1]), t, self.bigram_tc, self.trigram_tc, self.tag_len)






