import math
import operator as op
from functools import partial

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

class LinearModel(object):
    """general model building class
    provide Weights dictionary
    l2 norm regularization not applied to bias units denoted with 'b' in weights
    """
    def __init__(self, in_size, out_size, fl_act, lr=1e-1, reg=0,):
        
        self.in_size = in_size
        self.out_size = out_size
        self.fl_act = fl_act
        self.lr = lr
        self.reg = reg
        self.W = {'xy': self.init_weights(self.in_size, self.out_size),
                  'b': np.zeros((1, self.out_size,))} 
   
    def init_weights(self, *args):
        return np.random.randn(*args) * 0.01
        #return 2 * np.random.random(args) - 1
    
    def set_if_given(self, kwargs, key):
        if key not in kwargs:
            return self.W[key]
        else:
            return kwargs[key]
        
    def est_grad(self, X, y):
        def ind_W(W_key):
            W = self.W[W_key]
            ep = 1e-5
            grads = []
            for i in np.ndenumerate(W):
                ys = []
                for k in (op.add, op.sub):
                    if hasattr(self, 'reset'): self.reset()
                    W_mod = np.array(W, copy=True)
                    W_mod[i[0]] = k(i[1], ep)
                    pred = self.predict(X, **{W_key: W_mod})
                    loss = self.calc_loss(y, pred)
                    loss += self.calc_reg_loss(**{W_key: W_mod})
                    ys.append(loss)
                grad = np.subtract(*ys) / (2 * ep) 
                grads.append(grad)
            return np.array(grads).reshape(*W.shape)
        return {key: ind_W(key) for key in self.W.keys()}
    
    def calc_reg_loss(self, **kwargs):
        if not self.reg:
            return 0
        
        w = partial(self.set_if_given, kwargs)
        
        loss = 0
        for key in self.W.keys():
            if 'b' not in key:
                loss += np.sum(np.square(w(key))) * .5
        return loss * self.reg   
    
    def calc_dreg_loss(self):
        dW = {}
        for key in self.W.keys(): 
            if 'b' in key:
                dW[key] = np.zeros(self.W[key].shape)
            else:
                dW[key] = self.W[key] * self.reg
        return dW

    def predict(self, X, **kwargs):
        w = partial(self.set_if_given, kwargs)
        l1 = np.dot(X, w('xy')) + w('b')
        return self.fl_act.activate(l1)

    def calc_loss(self, y, pred):
        return self.fl_act.loss(y, pred)
        
    def calc_grad(self, X, y, pred):
        dpred = self.fl_act.dloss(y, pred)
        dW_xy = np.dot(X.T, dpred)
        dW_b = np.sum(dpred, axis=0, keepdims=True)
        return {'xy': dW_xy, 'b': dW_b}
    
    def step(self, X, y):
        pred = self.predict(X)
        
        loss = self.calc_loss(y, pred)
        loss += self.calc_reg_loss()
        
        grad = self.calc_grad(X, y, pred)
        grad = merge_with(sum, grad, self.calc_dreg_loss())
        grad = valmap(lambda x: x.clip(-5., 5.), grad)
        
        if not isinstance(self.lr, float):
            lr = self.lr.update(grad)
        else:
            lr = valmap(lambda x: self.lr, grad)

        for key in grad.keys():
            self.W[key] += grad[key] * -lr[key]

        return loss

class FeedFwd(LinearModel):
    
    def __init__(self, in_size, hidden_size, out_size, hl_act, fl_act, **kwargs):
        super(FeedFwd, self).__init__(in_size, out_size, fl_act, **kwargs)
        self.hidden_size = hidden_size
        self.hl_act = hl_act
        self.W = {'xl1': self.init_weights(self.in_size, self.hidden_size),
                  'bl1': np.zeros((1, self.hidden_size,)),
                  'l1y': self.init_weights(self.hidden_size, self.out_size),
                  'by': np.zeros((1, self.out_size,)),}
    
    def predict(self, X, **kwargs):
        w = partial(self.set_if_given, kwargs)
        l1 = self.hl_act(np.dot(X, w('xl1')) + w('bl1'))
        l2 = np.dot(l1, w('l1y')) + w('by')
        self.layers = [l1, l2]
        return self.fl_act.activate(l2)
        
    def calc_grad(self, X, y, pred):
        dpred = self.fl_act.dloss(y, pred)
        
        dW_l1y = np.dot(self.layers[0].T, dpred)
        dW_by = np.sum(dpred, axis=0, keepdims=True)
        
        d_l1 = np.dot(dpred, self.W['l1y'].T)
        d_l1 = d_l1 * self.hl_act(self.layers[0], deriv=True)
        
        dW_xl1 = np.dot(X.T, d_l1)
        dW_b1 = np.sum(d_l1, axis=0, keepdims=True)
        
        return {'xl1': dW_xl1, 'bl1': dW_b1, 'l1y': dW_l1y, 'by': dW_by}

class RNN(FeedFwd):
    
    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs.pop('batch_size', 1)
        super(RNN, self).__init__(*args, **kwargs)
        self.reset()
        self.W = {'xh': self.init_weights(self.in_size, self.hidden_size),
                  'hh': self.init_weights(self.hidden_size, self.hidden_size), 
                  'bh': np.zeros((1, self.hidden_size,)),    
                  'hy': self.init_weights(self.hidden_size, self.out_size),
                  'by': np.zeros((1, self.out_size,)),}
    
    def reset(self):
        self.hs = [np.zeros([self.batch_size, self.hidden_size])]
        self.dh = np.zeros([self.batch_size, self.hidden_size])
    
    def _predict(self, X, **kwargs):    
        w = partial(self.set_if_given, kwargs)        
        terms = [np.dot(self.hs[-1], w('hh')),
                 np.dot(X, w('xh')),
                 w('bh')]
        h = self.hl_act(reduce(op.add, terms))
        l2 = np.dot(h, w('hy')) + w('by')
        
        self.hs.append(h)         
        return self.fl_act.activate(l2)
    
    def predict(self, Xs, **kwargs):
        return map(lambda x: self._predict(x, **kwargs), Xs)
    
    def calc_loss(self, ys, preds):
        losses = map(lambda x: self.fl_act.loss(*x), zip(ys, preds))
        return sum(losses) / len(losses)
    
    def _calc_grad(self, X, y, pred, i):
        dpred = self.fl_act.dloss(y, pred)
        
        dw_hy = np.dot(self.hs[i].T, dpred)
        dw_yb = np.sum(dpred, axis=0, keepdims=True)
        
        d_l1 = np.dot(dpred, self.W['hy'].T) + self.dh
        d_l1 = d_l1 * self.hl_act(self.hs[i], deriv=True)
        
        dw_xh = np.dot(X.T, d_l1)
        dw_hh = np.dot(self.hs[i-1].T, d_l1)
        dw_hb = np.sum(d_l1, axis=0, keepdims=True)
        
        self.dh = np.dot(d_l1, self.W['hh'].T)

        #d_l0 = np.dot(d_l1, self.W['xh'].T)
        
        return {'xh': dw_xh, 'hh': dw_hh, 'bh': dw_hb, 'hy': dw_hy, 'by': dw_yb}
    
    def calc_grad(self, Xs, ys, preds):
        inds = reversed(range(len(Xs)))
        grads = map(lambda ind: self._calc_grad(Xs[ind], ys[ind], preds[ind], ind+1), inds)
        grads_sum = reduce(partial(merge_with, sum), grads)
        return valmap(lambda x: x / len(Xs), grads_sum)
        
    def step(self, *args, **kwargs):
        loss = super(RNN, self).step(*args, **kwargs) 
        self.reset()
        return loss