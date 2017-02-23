from functools import partial

import numpy as np
from sklearn import preprocessing
from toolz.dicttoolz import itemmap
from toolz import compose


class Encoder(object):
    def __init__(self, text):        
        self.text = text.lower()
        self.chars = list(set(self.text))
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.chars)
        
        self.inds = {self.le.transform(i): i for i in self.chars}
        
        self.enc = preprocessing.OneHotEncoder()
        classes = self.le.transform(self.le.classes_)
        self.enc.fit(classes.reshape(-1,1))
    
    def to_ind(self, char):
        return self.le.transform(char)
    
    def to_vect(self, char):
        ind = self.to_ind(char)
        return self.enc.transform(ind).toarray()[0]

class SpacyEncoder(object):
    def __init__(self, nlp, text):
        self.nlp = nlp
        text = self.nlp(unicode(text))
    
        summary = 'total word count:\t{0}'.format(len(text))

        mapped_count = len([i for i in text if i.has_vector])
        summary += '\nmapped word count:\t{0}'.format(mapped_count)
                
        unique_words = {i.text for i in text}
        self.word_count = len(unique_words)
        summary += '\nnum unique words:\t{0}'.format(self.word_count)
        print(summary)

        self.lookup = {i: ind for ind, i in enumerate(unique_words)}
        self.inds = itemmap(reversed, self.lookup)
        self.text = text
        
    def to_vect(self, word):
        ind = self.nlp.vocab.strings[word]
        return self.nlp.vocab[ind].vector

    def ind_to_target_vect(self, ind):
        v = np.zeros(self.word_count)
        v[ind] = 1.
        return v
       

def gen_sample(mod, enc, seed_val, sequence_len, sep=''):
    mod.reset()
    ys = [seed_val,]
    for i in range(sequence_len):
        inp = enc.to_vect(ys[-1])
        pred = mod._predict(inp)[0]
        ind = np.random.choice(range(len(pred)), p=pred.ravel())
        ys.append(enc.inds[ind])

    return sep.join(ys)


MISSINGWORD = 'MISSINGWORD'

def drop_unkown_and_rare_words(nlp, text):
    counts = {}
    doc = nlp.make_doc(unicode(text))
    for i in doc:
        ind = nlp.vocab.strings[i.text]
        try:
            counts[ind] += 1
        except:
            counts[ind] = 1
    
    text_clean = ''
    for i in doc:    
        ind = nlp.vocab.strings[i.text] 
        too_infrequent = counts[ind] < 2
        conds = (too_infrequent, i.is_oov)
        text_clean += MISSINGWORD if any(conds) else i.text
        text_clean += str(i.whitespace_)
        
    return text_clean

def encode_inp(enc, text):
    inp = enc.nlp(unicode(text))
    inp = [i.vector for i in inp]
    return np.array([inp])

def fill_sequence(sequence_len, Xs):
    Xs = Xs[:,-sequence_len:,:] # truncate too long of inputs
    sequence_missing = sequence_len - Xs.shape[1]
    if sequence_missing:
        shape = list(Xs.shape)
        shape[1] = sequence_missing
        padding = np.zeros(shape)
        Xs = np.concatenate([padding, Xs], axis=1)
    return Xs

def sample_next(enc, mod, Xs):
    pred = mod.predict(Xs)[0]
    for i in range(5):
        ind = np.random.choice(range(len(pred)), p=pred.ravel())
        word = enc.inds[ind]
        if word != MISSINGWORD:
            break
    vect = enc.to_vect(word)
    Xs_ext = np.concatenate([Xs, np.array([[vect]])], axis=1)
    return word, vect, Xs_ext
    
def gen_string(enc, mod, sequence_len, text, n):
    Xs = encode_inp(enc, text.lower())
    fs = compose(partial(sample_next, enc, mod), 
                 partial(fill_sequence, sequence_len))
    ys = []
    for i in range(n):
        y, vect, Xs = fs(Xs)
        ys.append(y)
    return text + ' ' + ' '.join(ys)

def data_gen(enc, sequence_len, batch_size):  
    inds = range(len(enc.text)-sequence_len-1)
    fs = compose(enc.ind_to_target_vect, lambda x: enc.lookup[x.text])   
    while True:
        sel_inds = np.random.choice(inds, size=batch_size, replace=False)
        Xs, ys = [], []
        for i in sel_inds:
            row = enc.text[i:i+sequence_len]
            X = map(lambda x: x.vector, row)
            Xs.append(X)           
            ys.append(fs(enc.text[i+1]))
        yield map(np.array, (Xs, ys))