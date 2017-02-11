import numpy as np


def relu(x, deriv=False):
    if deriv:
        f = np.vectorize(lambda x: 0. if x <= 0. else 1.)
        return f(x)
    return np.maximum(0, x)

def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def tanh(x, deriv=False):
    if deriv:
        return 1 - x**2
    return np.tanh(x)

class MSEReg(object):
    
    def activate(self, x):
        return x
    
    def loss(self, act, pred):
        errs = pred - act
        f = np.vectorize(lambda x: x ** 2)
        return f(errs).sum() / act.shape[0]

    def dloss(self, act, pred):
        errs = pred - act
        return errs * (2. / act.shape[0])
    
class Softmax(object):
    
    def activate(self, x):
        tf = np.exp(x)
        return tf / np.sum(tf, axis=1, keepdims=True)
    
    def loss(self, act, pred):
        """calculates the correct log prob loss"""
        epsilon = 1e-15
        pred = np.maximum(epsilon, pred)
        pred = np.minimum(1-epsilon, pred)
        ll = -np.log(pred[range(len(act)), act])
        return sum(ll) / len(ll)
        
    def dloss(self, act, pred):
        dpred = pred
        dpred[range(len(act)),act] -= 1
        return dpred / len(dpred)