import operator as op
from functools import partial

import numpy as np
from toolz import accumulate
from toolz.dicttoolz import merge_with, valmap


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
    
    def _calc_reg_loss(self, **kwargs):
        if not self.reg:
            return 0
        
        w = partial(self.set_if_given, kwargs)
        
        loss = 0
        for key in self.W.keys():
            if 'b' not in key:
                loss += np.sum(np.square(w(key))) * .5
        return loss * self.reg 

    def calc_loss(self, y, pred, **kwargs):
        loss = self.fl_act.loss(y, pred)
        loss += self._calc_reg_loss(**kwargs)
        return loss  
    
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
        
        try:
            l1 = self.fl_act.activate(l1)
        except:
            l1 = self.fl_act(l1)

        self.layers = [X, l1,]
        return l1
        
    def calc_dpred(self, y, pred):
        return self.fl_act.dloss(y, pred)

    def _calc_W_grad(self, dpred):
        try:
            dpred *= self.fl_act(self.layers[1], deriv=True)
        except:
            pass

        dW_xy = np.dot(self.layers[0].T, dpred)
        dW_b = np.sum(dpred, axis=0, keepdims=True)
        
        self.dX = np.dot(dpred, self.W['xy'].T)
        return {'xy': dW_xy, 'b': dW_b}

    def calc_grad(self, dpred):
        grad = self._calc_W_grad(dpred)
        grad = merge_with(sum, grad, self.calc_dreg_loss())
        return valmap(lambda x: x.clip(-5., 5.), grad)

    def est_grad(self, X, y):
        def ind_W(W_key):
            W = self.W[W_key]
            ep = 1e-5
            grads = []
            for i in np.ndenumerate(W):
                losses = []
                for k in (op.add, op.sub):
                    if hasattr(self, 'reset'): self.reset()
                    W_mod = np.array(W, copy=True)
                    W_mod[i[0]] = k(i[1], ep)
                    pred = self.predict(X, **{W_key: W_mod})
                    loss = self.calc_loss(y, pred, **{W_key: W_mod})
                    losses.append(loss)
                grad = np.subtract(*losses) / (2 * ep) 
                grads.append(grad)
            return np.array(grads).reshape(*W.shape)
        return {key: ind_W(key) for key in self.W.keys()}

    def update(self, grad):
        if not isinstance(self.lr, float):
            lr = self.lr.update(grad)
        else:
            lr = valmap(lambda x: self.lr, grad)

        for key in grad.keys():
            self.W[key] += grad[key] * -lr[key]
    
    def step(self, X, y):
        pred = self.predict(X)
        loss = self.calc_loss(y, pred)
        dpred = self.calc_dpred(y, pred)
        grad = self.calc_grad(dpred)
        self.update(grad)
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
        
        try:
            l2 = self.fl_act.activate(l2)
        except:
            l2 = self.fl_act(l2)


        self.layers = [X, l1, l2]
        return l2
        
    def _calc_W_grad(self, dpred):
        try:
            dpred *= self.fl_act(self.layers[2], deriv=True)
        except:
            pass
        
        dW_l1y = np.dot(self.layers[1].T, dpred)
        dW_by = np.sum(dpred, axis=0, keepdims=True)
        
        d_l1 = np.dot(dpred, self.W['l1y'].T)
        d_l1 *= self.hl_act(self.layers[1], deriv=True)
        
        dW_xl1 = np.dot(self.layers[0].T, d_l1)
        dW_b1 = np.sum(d_l1, axis=0, keepdims=True)
        
        self.dX = np.dot(d_l1, self.W['xl1'].T)
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
        self.layers = [[None, np.zeros([self.batch_size, self.hidden_size]),None]]
        self.dh = np.zeros([self.batch_size, self.hidden_size])
        self.dX = []
    
    def _predict(self, X, **kwargs):    
        w = partial(self.set_if_given, kwargs)        
        terms = [np.dot(self.layers[-1][1], w('hh')),
                 np.dot(X, w('xh')),
                 w('bh')]
        l1 = self.hl_act(reduce(op.add, terms))
        l2 = np.dot(l1, w('hy')) + w('by')       
        
        try:
            l2 = self.fl_act.activate(l2)
        except:
            l2 = self.fl_act(l2)
        
        self.layers.append([X, l1, l2])
        return l2
    
    def predict(self, Xs, **kwargs):
        return map(lambda x: self._predict(x, **kwargs), Xs)
    
    def calc_loss(self, ys, preds, **kwargs):
        losses = map(lambda x: self.fl_act.loss(*x), zip(ys, preds))
        loss = sum(losses) / len(losses)
        loss += self._calc_reg_loss(**kwargs)
        return loss
    
    def _calc_ind_W_grad(self, dpred, i):
        i += 1
        try:
            dpred *= self.fl_act(self.layers[i][2], deriv=True)
        except:
            pass
        
        dw_hy = np.dot(self.layers[i][1].T, dpred)
        dw_yb = np.sum(dpred, axis=0, keepdims=True)
        
        d_l1 = np.dot(dpred, self.W['hy'].T) + self.dh
        d_l1 *= self.hl_act(self.layers[i][1], deriv=True)
        
        dw_xh = np.dot(self.layers[i][0].T, d_l1)
        dw_hh = np.dot(self.layers[i-1][1].T, d_l1)
        dw_hb = np.sum(d_l1, axis=0, keepdims=True)
        
        self.dh = np.dot(d_l1, self.W['hh'].T)

        dX = np.dot(d_l1, self.W['xh'].T)
        self.dX.append(dX)
        
        return {'xh': dw_xh, 'hh': dw_hh, 'bh': dw_hb, 'hy': dw_hy, 'by': dw_yb}

    def calc_dpred(self, ys, preds):
        return map(lambda x: self.fl_act.dloss(*x), zip(ys, preds))

    def _calc_W_grad(self, dpreds):
        inds = reversed(range(len(dpreds)))
        grads = map(lambda ind: self._calc_ind_W_grad(dpreds[ind], ind), inds)
        self.dX = list(reversed(self.dX))
        grads_sum = reduce(partial(merge_with, sum), grads)
        return valmap(lambda x: x / len(dpreds), grads_sum)
        
    def step(self, *args, **kwargs):
        loss = super(RNN, self).step(*args, **kwargs) 
        self.reset()
        return loss

class StackedModels(object):
    """check connecting widths are correct, 
       check that final layers on intermediate models are not summary layers
    """
    def __init__(self, mods):
        self.mods = mods

    def reset(self,):
        for i in self.mods:
            if hasattr(i, 'reset'):
                i.reset()
        
    def predict(self, X, layers=False):
        calc_pred = lambda f: f(lambda x, m: m.predict(x), self.mods, X)
        return calc_pred(accumulate) if layers else calc_pred(reduce)
    
    def _calc_reg_loss(self):
        return sum(map(lambda x: x._calc_reg_loss(), self.mods))

    def calc_loss(self, y, pred):
        return self.mods[-1].calc_loss(y, pred) + self._calc_reg_loss()
        
    def calc_dpred(self, y, pred):
        return self.mods[-1].calc_dpred(y, pred)

    def calc_grad(self, dpred):
        grads = {}
        for i in reversed(range(len(self.mods))):
            grad = self.mods[i].calc_grad(dpred)
            grad = merge_with(sum, grad, self.mods[i].calc_dreg_loss())
            grad = valmap(lambda x: x.clip(-5., 5.), grad)
            grads[i] = grad
            dpred = self.mods[i].dX
        return grads
    
    def est_grad(self, X, y):
        ep = 1e-5
        grads = []
        for i in np.ndenumerate(X):
            losses = []
            for k in (op.add, op.sub):
                #if hasattr(self, 'reset'): self.reset()
                X_mod = np.array(X, copy=True)
                X_mod[i[0]] = k(i[1], ep)
                pred = self.predict(X_mod)
                loss = self.calc_loss(y, pred)
                losses.append(loss)
            grad = np.subtract(*losses) / (2 * ep) 
            grads.append(grad)
        grads = np.array(grads).reshape(*X.shape)
        return {0: grads}

    def step(self, X, y):
        pred = self.predict(X)
        loss = self.calc_loss(y, pred)
        dpred = self.calc_dpred(y, pred)
        grads = self.calc_grad(dpred)
        for i in range(len(self.mods)):
            self.mods[i].update(grads[i])
            if hasattr(self.mods[i], 'reset'): self.mods[i].reset()
        return loss