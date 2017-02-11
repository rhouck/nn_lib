import math
import operator as op
from functools import partial

import numpy as np
from toolz.dicttoolz import merge_with, valmap


class adagrad_lr(object):
    def __init__(self, lr):
        self.lr = lr
        self.ss_grad = None       
        f = lambda x: self.lr / math.sqrt(x + 1e-8)
        self.f = np.vectorize(f)
        
    def update(self, grad):
        if not isinstance(self.ss_grad, dict):
            self.ss_grad = valmap(np.zeros_like, grad)
        
        self.ss_grad = merge_with(sum, self.ss_grad, valmap(np.square, grad))
        return valmap(self.f, self.ss_grad)

class BaseModel(object):
    """general model building class
    provide Weights dictionary
    l2 norm regularization not applied to bias units denoted with 'b' in weights
    """
    def __init__(self, final_layer, lr=1e-1, reg=0,):
        self.fl = final_layer
        self.lr = lr
        self.reg = reg
        self.W = {'None': None}
    
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
                    W_mod = np.array(W, copy=True)
                    W_mod[i[0]] = k(i[1], ep)
                    pred = self.predict(X, **{W_key: W_mod})
                    loss = self.fl.loss(y, pred)
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
    
    def step(self, X, y):
        pred = self.predict(X)
        
        loss = self.fl.loss(y, pred)
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
