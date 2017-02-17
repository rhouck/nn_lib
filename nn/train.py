import math

import numpy as np
from toolz import compose
from toolz.dicttoolz import merge_with, valmap


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
