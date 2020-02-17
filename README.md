## HMMs
### Deliverables - 
1. Bigram HMMs
2. Trigram HMMs
3. Evaluations

### Assumptions Made
1. Tokens with less than 1 count is converted to \<UNK\>.  The POS tag for those tokens was kept the same.
2. Smoothing- trying out k-smoothing.
3. emojis, urls, hashtags, mentions, numbers, dates, etc. were converted to a special token.
  
### How to run this
First, we need to load the corpus and tokenize.  This can be done using `TextProcess.py`
```
From TextProcess import ProcessedCorpus
corpus = ProcessedCorpus("CSE517_HW_HMM_Data/twt.bonus.json",\
                         "CSE517_HW_HMM_Data/twt.dev.json",\
                         "CSE517_HW_HMM_Data/twt.test.json",1)
```

Then the processed corpus is loaded to a language model using `LanguageModel.py`
```
From LanguageModel import LanguageModel
lm = LanguageModel(corpus)
```

`LanguageModel` class contains unigram, bigram, and trigram probabilities and count dictionaries for HMMs to use.  The smoothing options available are `add-k` and linear interpolation.  The smoothing parameters are updated and the necessary calculation updates to probabilities can be done as below:

```
lm.update(0.3,(0.00001,0.99999),(0.001,0.001,0.998))
```
The Bigram and Trigram HMMs are implemented in `BigramHMM.py` and `TrigramHMM.py`, and these can run as follows:
```
bhmm = BigramHMM(corpus, lm)
pis, preds = bhmm.test_viterbi(bhmm.dev, bhmm.dev_emission_probabilities)
accuracy, y_true, y_pred, confusion_matrix_array, normalized= bhmm.analyze_results(bhmm.dev, preds)
```
```
thmm = TrigramHMM(corpus, lm)
pis, preds = thmm.test_trigram_viterbi(thmm.dev, thmm.dev_emission_probabilities)
accuracy, y_true, y_pred, confusion_matrix_array, normalized= thmm.analyze_results(thmm.dev, preds)
```

The confusion matrix can be plotted from above output, `normalized` and `confusion_matrix_array`.
```
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_cm = pd.DataFrame(normalized, index = list(bhmm.tags),
                  columns = list(bhmm.tags))

plt.figure(figsize = (40,40))

ax = sn.heatmap(df_cm, annot=True, cbar=False)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
sn.set(font_scale=1.9)
```
