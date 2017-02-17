import numpy as np
from sklearn import preprocessing


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

def gen_sample(mod, enc, seed_val, sequence_len):
    mod.reset()
    ys = [seed_val,]
    for i in range(sequence_len):
        inp = enc.to_vect(ys[-1])
        pred = mod._predict(inp)[0]
        ind = np.random.choice(range(len(pred)), p=pred.ravel())
        ys.append(enc.inds[ind])

    return ''.join(ys)