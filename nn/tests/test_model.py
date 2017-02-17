import unittest

import numpy as np

from nn.utils import *
from nn.activation import *
from nn.model import *
from nn.train import *
from nn.nlp import *


mean = lambda l: sum(l) / float(len(l))

class TestLinearModel(unittest.TestCase):

    def test_gradient_calc(self):
        # generate regression data
        in_width = 3
        out_width = 2
        X = np.random.randn(200, in_width)
        W_xy_true = np.random.randn(in_width, out_width)
        y = np.dot(X, W_xy_true)

        h = LinearModel(in_width, out_width, MSEReg())
        pred = h.predict(X)
        dpred = h.calc_dpred(y, pred)
        grad = h.calc_grad(dpred)
        est_grad = h.est_grad(X, y)
        ratios = check_grads(grad, est_grad).values()
        self.assertTrue(mean(ratios)  > .9)

class TestFeedFwdModel(unittest.TestCase):

    def setUp(self):
        # generate classification data
        N = 100
        D = 2 # dimensionality
        K = 3 # number of classes
        X = np.zeros((N*K,D)) # data matrix (each row = single example)
        y = np.zeros(N*K, dtype='uint8') # class labels
        for j in xrange(K):
            ix = range(N*j,N*(j+1))
            r = np.linspace(0.0,1,N) # radius
            t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
            X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            y[ix] = j
        self.in_size = D
        self.out_size = K
        self.X = X
        self.y = y
        
    def test_gradient_calc(self):
        hidden_size = 10
        h = FeedFwd(self.in_size, hidden_size, self.out_size, relu, Softmax())
        pred = h.predict(self.X)
        dpred = h.calc_dpred(self.y, pred)
        grad = h.calc_grad(dpred)
        est_grad = h.est_grad(self.X, self.y)
        ratios = check_grads(grad, est_grad).values()
        self.assertTrue(mean(ratios)  > .9)

class TestStackedLinearModel(TestFeedFwdModel):

    def test_gradient_calc(self):
        hidden_size = 10
        mods = [LinearModel(self.in_size, hidden_size, relu),
                LinearModel(hidden_size, self.out_size, Softmax())]
        h = StackedModels(mods)
        
        data = get_minibatch(self.X, self.y)
        stats = train(h, data, 1, nepochs=50, check=False)
        
        h.step(self.X, self.y)
        act = {0: h.mods[0].dX}
        est = h.est_grad(self.X, self.y)
        ratios = check_grads(act, est, rel_error_thresh=1e-1).values()
        self.assertTrue(mean(ratios)  > .9)

class TestRNN(unittest.TestCase):

    def test_gradient_calc(self):
        text = "let's see if you can learn this sentance"
        enc = Encoder(text)
        X, y = [], []
        for i in range(len(text)):
            if i:
                X.append(enc.to_vect(text[i-1]))
                y.append(enc.to_ind(text[i]))
        X, y = map(np.array, (X, y))

        in_size = X.shape[1]
        out_size = X.shape[1]
        hidden_size = 5
        sequence_len = 5
        batch_size = 5
        h = RNN(in_size, hidden_size, out_size, relu, Softmax(), batch_size=batch_size)

        data = get_seq_minibatch(X, y, batch_size, sequence_len,)
        Xys = data.next()
        Xs = map(first, Xys)
        ys = map(second, Xys)
        _ = h.step(map(first, Xys), map(second, Xys))                      

        Xys = data.next()
        Xs = map(first, Xys)
        ys = map(second, Xys)
        preds = h.predict(Xs)
        loss = h.calc_loss(ys, preds)
        dpreds = h.calc_dpred(ys, preds)
        grad = h.calc_grad(dpreds)
        est_grad = h.est_grad(Xs, ys)
        ratios = check_grads(grad, est_grad).values()
        self.assertTrue(mean(ratios)  > .9)