import json
from operator import add

import numpy as np
from toolz import first, second, accumulate
from toolz.dicttoolz import merge, merge_with, valmap


def accuracy(y, pred):
    if all(map(lambda x: isinstance(x, list), (y, pred))):
        accs = map(lambda x: accuracy(*x), zip(y, pred))
        return sum(accs) / float(len(accs))
    f = np.vectorize(lambda x: True if x == 0 else False)
    correct = f(np.argmax(pred, axis=1) - y)
    return sum(correct) / float(len(correct))

def check_grads(grad, est_grad, rel_error_thresh=1e-2):  
    f = np.vectorize(lambda x: 1 if x < rel_error_thresh else 0)   
    checks = {}
    for key in grad.keys():
        abs_diff = abs(grad[key] - est_grad[key])
        sum_grads = abs(grad[key]) + abs(est_grad[key]) 
        rel_error = abs_diff / np.maximum(1e-10, sum_grads)
        checks[key] =  round(f(rel_error).sum() / float(rel_error.size), 2) 
    return checks

def get_minibatch(X, y, batch_size=None):
    batch_size = len(X) if not batch_size else batch_size
    while True:
        inds = np.random.choice(np.arange(len(X)), size=batch_size, replace=False)
        yield X[inds], y[inds]

def get_seq_minibatch(X, y, batch_size, sequence_len):    
    max_ind = len(X) - sequence_len - 1
    while True:
        inds = np.random.choice(np.arange(max_ind), size=batch_size, replace=False)
        inds_inc = accumulate(add, [inds] + [1] * (sequence_len - 1))        
        yield map(lambda inds: (X[inds], y[inds]), inds_inc)

def is_sequence_type(Xy):
    a = not isinstance(Xy[0], np.ndarray)
    b = isinstance(Xy[0][0], np.ndarray)
    return all([a, b])

def split_Xy(Xy):
    if not is_sequence_type(Xy):
        X, y = Xy
    else:
        X = map(first, Xy)
        y = map(second, Xy)
    return X, y

def train(mod, data_gen, num_batch_per_epoch, nepochs=100, check=True):
    
    def get_acc(y, pred):
        try:
            return round(accuracy(y, pred), 2)
        except:
            return np.nan

    stats = []
    try:
        for epoch in range(nepochs):
            for batch in range(num_batch_per_epoch):
                Xy = data_gen.next()
                X, y = split_Xy(Xy)
                _ = mod.step(X, y)
            
            if epoch and not epoch % int((nepochs * .05)):
                                
                Xy = data_gen.next()
                X, y = split_Xy(Xy)
                pred = mod.predict(X)
                loss = mod.calc_loss(y, pred)
                acc = get_acc(y, pred)
                
                print('epoch {0}:\tloss: {1:0.5f}\tacc: {2}'.format(epoch, loss, acc))

                # grad = mod.calc_grad(X, y, pred)
                # grad = merge_with(sum, grad, mod.calc_dreg_loss())
                # f = lambda x: abs(x).mean()
                # grad_scale = valmap(f, grad)

                # print('epoch {0}:\tloss: {1:0.5f}\tacc: {2}'.format(epoch, loss, acc))
                # f = lambda x: '{0}: {1:.1e}'.format(*x)
                # print('\t\t' + ' '.join(map(f, grad_scale.items())))
                # stats.append(merge({'epoch': epoch, 'loss': loss, 'acc': acc}, grad_scale))

                # if check and epoch <= (nepochs * .2):
                #     est_grad = mod.est_grad(X, y)
                #     grad_acc = check_grads(grad, est_grad)
                #     print('grad accuracy:' + str(grad_acc))
                  
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