import json

import numpy as np
from toolz.dicttoolz import merge_with, valmap

def accuracy(mod, X, y):
    f = np.vectorize(lambda x: True if x == 0 else False)
    correct = f(np.argmax(mod.predict(X), axis=1) - y)
    return sum(correct) / float(len(correct))

def check_grads(grad, est_grad):
    
    f = np.vectorize(lambda x: 1 if x < 1e-2 else 0)
    
    checks = {}
    for key in grad.keys():
        abs_diff = abs(grad[key] - est_grad[key])
        sum_grads = abs(grad[key]) + abs(est_grad[key]) 
        rel_error = abs_diff / np.maximum(1e-10, sum_grads)
        checks[key] =  round(f(rel_error).sum() / float(rel_error.size), 2)
        #checks[key] = np.mean(rel_error) < 1e-2

        # non_zero_grad = np.maximum(1e-10, abs(grad[key]))
        # mean_diff_rat = np.mean(abs_diff / non_zero_grad)
        # checks[key] = mean_diff_rat < 1e-1
    return checks

# class MiniBatch(object):
#     def __init__(self, X, y, batch_size):
#         self.X = X
#         self.y = y
#         self.batch_size = batch_size

#     def num_batch_per_epoch():
#         return int(len(self.X) / self.batch_size)

#     def gen(self):
#         while True:
#             inds = np.random.choice(np.arange(len(self.X)), size=n, replace=False)
#             yield self.X[inds], self.y[inds]

# class SeqMiniBatch(MiniBatch):
#     def __init__(self, *args, **kwargs):
#         super(SeqMiniBatch, self).__init__(*args, **kwargs)
#         self.sequence_len = sequence_len

#     def gen(self):
#         count = 0
#         max_ind = len(self.X) - self.sequence_len - 1
#         while True:  
#             if not count:
#                 inds = np.random.choice(np.arange(max_ind), size=self.batch_size, replace=False)
#             else:
#                 inds += 1
            
#             count += 1
#             if count == self.sequence_len:
#                 count = 0
            
#             yield self.X[inds], self.y[inds]


def get_minibatch(X, y, n=None):
    if not n:
        n = len(X)
    while True:
        inds = np.random.choice(np.arange(len(X)), size=n, replace=False)
        #np.random.shuffle(inds)
        yield X[inds], y[inds]

def get_seq_minibatch(X, y, batch_size, sequence_len):    
    count = 0
    max_ind = len(X) - sequence_len - 1
    while True:  
        if not count:
            inds = np.random.choice(np.arange(max_ind), size=batch_size, replace=False)
        else:
            inds += 1
        
        count += 1
        if count == sequence_len:
            count = 0
        
        yield X[inds], y[inds]


def train(mod, data_gen, num_batch_per_epoch, nepochs=100, check=True, sequence_len=None):
    
    def grad_accuracy(mod, X, y):
        pred = mod.predict(X)
        grad = mod.calc_grad(X, y, pred)
        grad = merge_with(sum, grad, mod.calc_dreg_loss())
        est_grad = mod.est_grad(X, y)
        return check_grads(grad, est_grad) 

    def get_acc(mod, X, y):
        try:
            return round(accuracy(mod, X, y), 2)
        except:
            return np.nan

    stats = []
    try:
        for epoch in range(nepochs):
            for batch in range(num_batch_per_epoch):
        
                if not sequence_len:
                    X, y = data_gen.next()
                    loss = mod.step(X, y)
                else:
                    for step in range(sequence_len):
                        X, y = data_gen.next()
                        _ = mod.step(X, y)
            
            # calc performance
            if epoch and not epoch % int((nepochs * .1)):
                
                if not sequence_len:
                    acc = get_acc(mod, X, y)
                    if check and epoch <= (nepochs * .2):                
                        print grad_accuracy(mod, X, y)
                else:
                    loss, acc =  0, 0
                    for step in range(sequence_len):
                        X, y = data_gen.next()
                        if check and step == 1 and epoch <= (nepochs * .2):                
                           print grad_accuracy(mod, X, y)      

                        loss += mod.step(X, y)
                        acc += get_acc(mod, X, y)

                    loss /= float(sequence_len)
                    acc /= float(sequence_len)

                stats.append((epoch, loss, acc))
                print('epoch {0}\tloss: {1:0.5f}\tacc: {2}'.format(*stats[-1]))
                
    except KeyboardInterrupt as err:
        print('stopping optimzation')

    return stats

def write_weights(fn, W):
    weights = valmap(lambda x: x.tolist(), W)
    with open(fn, 'w') as f:
        json.dump(weights, f)
        
def load_weights(fn):
    with open(fn, 'r') as f:
        weights = json.load(f)
        return valmap(np.array, weights)