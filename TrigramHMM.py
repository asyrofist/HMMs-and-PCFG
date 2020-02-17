from TextProcess import ProcessedCorpus
from LanguageModel import LanguageModel
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix

# For TrigramHMM class, the two inputs are ProcessedCorpus class, and LanguageModel class
# examples:
# corpus = ProcessedCorpus("CSE517_HW_HMM_Data/twt.train.json",\
#                          "CSE517_HW_HMM_Data/twt.dev.json",\
#                          "CSE517_HW_HMM_Data/twt.test.json",1)
#
# lm = LanguageModel(corpus)
# lm.update(0.3,(0.00001,0.99999),(0.001,0.019,0.98))

class TrigramHMM:
    def __init__(self, corpus, lm):
        self.corpus = corpus
        self.lm = lm

        # corpuses for tuning + testing
        self.dev = lm.dev
        self.test = lm.test
        self.tags = corpus.tag_vocab

        # emission probability calculation
        self.k = 1 # emission probability smoothing parameter
        self.dev_emission_probabilities = self.emission_proba_dict(self.dev)
        self.test_emission_probabilities = self.emission_proba_dict(self.test)

        # transition probability
        self.trigram_transition_proba = lm.trigram_probabilities

        # initial transition proba distribution
        self.initial = defaultdict(float)
        self.initial_distribution()

    def emission_proba(self, token):
        unigram_count = self.lm.unigram_count
        unigram_tc = self.lm.unigram_tc
        word, tag = token
        if (word, tag) not in unigram_count:
            p = self.k/(unigram_tc.get(tag,0) + self.k * (len(unigram_tc) + len(unigram_count)))
        else:
            p = (unigram_count.get((word,tag))+self.k)/(unigram_tc.get(tag,0) + self.k * (len(unigram_tc) + len(unigram_count)))
        return p

    def emission_proba_dict(self, dataset):
        emission_probabilities = defaultdict(float)
        for sentence in dataset:
            for word, _ in sentence:
                for tag in self.tags:
                    emission_probabilities[(word, tag)] = self.emission_proba((word, tag))
        return emission_probabilities

    def initial_distribution(self):
        for t in self.tags:
            self.initial[("<START>","<START>",t)] = self.lm.trigram_tc[("<START>","<START>",t)]/len(self.lm.train)



    def trigram_viterbi(self, sentence, emission_proba):
        pi = defaultdict(dict)
        # initialize
        for t in self.tags:
            pi[(0, sentence[0][0])][t] = {"prob" : self.initial[("<START>","<START>",t)] * emission_proba[(sentence[0][0],t)],\
                          "prev": "<START>"}

        # go through the rest of the sentence
        for i in range(1,len(sentence)):
            prev_word = sentence[i-1][0]
            word = sentence[i][0]
            for prev2 in self.tags:
                for current_tag in self.tags:
                    proba = dict()
                    for prev1 in self.tags:
                        p = self.trigram_transition_proba[(prev1, prev2, current_tag)] * emission_proba[(word, current_tag)]
                        proba[prev1] = p* pi[(i-1, prev_word)][prev1]["prob"]
                    prev_state = max(proba, key=proba.get)
                    max_proba = proba[prev_state]
                    pi[(i,word)][current_tag] = {"prob" : max_proba, "prev" : prev_state}
        return pi

    def backtrace(self, pi, sentence):
        sentence = [(i,x[0]) for i,x in enumerate(sentence)]

        final_word = sentence[-1]
        final_tag = None
        max_p = 0-

        for tag, d in pi[final_word].items():
            if d["prob"] > max_p:
                final_tag = tag-
                max_p = d["prob"]
                prev_state = d["prev"]

        pred_tags = [final_tag]
        for word in sentence[::-1]:
            final_tag = pi[word][final_tag]["prev"]
            pred_tags.append(final_tag)
        return pred_tags[::-1]

    def analyze_results(self, testset, preds):
        y_true = []

        for sentence in testset:
            for word in sentence:
                y_true.append(word[1])

        y_pred = []
        for sentence in preds:
            for tag in sentence:
                y_pred.append(tag)

        c = 0 # number of correct
        N = 0 # total number of predictions
        for i in range(len(y_pred)):
             N += 1
             if y_pred[i] == y_true[i]:
                 c += 1
        accuracy = c/N

        confusion_matrix_array = confusion_matrix(y_true, y_pred, labels = list(self.tags))
        normalized = confusion_matrix_array / (confusion_matrix_array.astype(np.float).sum(axis=1)+0.01)
        return accuracy, y_true, y_pred, confusion_matrix_array, normalized







