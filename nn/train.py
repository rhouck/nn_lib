import math
from operator import add

import numpy as np
from matplotlib import pyplot as plt
from toolz import first, second, accumulate, compose
from toolz.dicttoolz import merge, merge_with, keymap, valmap, itemmap


class adagrad_lr(object):
    def __init__(self, lr, max_adjusted_lr=10):
        self.lr = lr    
        self.max_adjusted_lr = max_adjusted_lr   
        f = lambda x: self.lr / math.sqrt(x + 1e-8)
        self.f = np.vectorize(f)
        
    def update(self, grad):
        if not hasattr(self, 'ss_grad'):
            self.ss_grad = valmap(np.zeros_like, grad)
        self.ss_grad = merge_with(sum, self.ss_grad, valmap(np.square, grad))
        f = compose(lambda x: np.minimum(x, self.max_adjusted_lr), self.f)
        return valmap(f, self.ss_grad)

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

def accuracy(y, pred):
    if all(map(lambda x: isinstance(x, list), (y, pred))):
        accs = map(lambda x: accuracy(*x), zip(y, pred))
        return sum(accs) / float(len(accs))
    f = np.vectorize(lambda x: True if x == 0 else False)
    correct = f(np.argmax(pred, axis=1) - y)
    return sum(correct) / float(len(correct))

def combine(grad):
    prepend_key = lambda x, d: keymap(lambda k: '{0}_{1}'.format(x, k), d)
    grad = itemmap(lambda i: (i[0], prepend_key(i[0], i[1])), grad)
    return merge(grad.values())

def train(mod, data_gen, num_batch_per_epoch, nepochs=100, check=True):
    
    def try_acc(y, pred):
        try:
            return round(accuracy(y, pred), 2)
        except:
            return np.nan

    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

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
                acc = try_acc(y, pred)
                
                dpred = mod.calc_dpred(y, pred)
                grad = mod.calc_grad(dpred)
                
                f = lambda x: abs(x).mean()
                try:
                    grad_mean = valmap(f, grad)
                except:
                    grad = combine(grad)
                    grad_mean = valmap(f, grad)

                stats.append(merge({'epoch': epoch, 'loss': loss, 'acc': acc}, grad_mean))
                
                print('epoch {0}:\tloss: {1:0.5f}\tacc: {2}'.format(epoch, loss, acc))
                grad_mean_groups = chunks(grad_mean.items(), 5)
                f = lambda x: '{0}: {1:.1e}'.format(*x)
                for grad_means in grad_mean_groups:
                    print('\t\t' + ' '.join(map(f, grad_means)))
                

                # if check and epoch <= (nepochs * .2):
                #     est_grad = mod.est_grad(X, y)
                #     grad_acc = check_grads(grad, est_grad)
                #     print('grad accuracy:' + str(grad_acc))
                  
    except KeyboardInterrupt as err:
        print('stopping optimzation')
    
    return stats

def plot_stats(stats):
    fig, ax = plt.subplots(ncols=3, figsize=[16,3])

    get_col = lambda stats, c: map(lambda x: x[c], stats)
    X = get_col(stats, 'epoch')
    perf_cols = ['loss', 'acc']
    for ind, col in enumerate(perf_cols):
        y = get_col(stats, col)
        ax[ind].plot(X, y)
        ax[ind].set_title(col)

    grad_cols = [i for i in stats[0].keys() if i not in perf_cols + ['epoch']]

    for ind, col in enumerate(grad_cols):
        y = get_col(stats, col)
        ax[2].plot(X, y, label=col)
    ax[2].legend(loc="upper right")
    ax[2].set_title('gradients')