import json

import numpy as np
from toolz import first, second, accumulate
from toolz.dicttoolz import valmap


def check_grads(grad, est_grad, rel_error_thresh=1e-2):  
    f = np.vectorize(lambda x: 1 if x < rel_error_thresh else 0)   
    checks = {}
    for key in grad.keys():
        abs_diff = abs(grad[key] - est_grad[key])
        sum_grads = abs(grad[key]) + abs(est_grad[key]) 
        rel_error = abs_diff / np.maximum(1e-10, sum_grads)
        checks[key] =  round(f(rel_error).sum() / float(rel_error.size), 2) 
    return checks

def write_weights(fn, W):
    def remove_numpy(W):
        if isinstance(W, list):
            return [remove_numpy(i) for i in W]
        return valmap(lambda x: x.tolist(), W)
    
    with open(fn, 'w') as f:
        json.dump(remove_numpy(W), f)
        
def load_weights(fn):
    with open(fn, 'r') as f:
        weights = json.load(f)
        return valmap(np.array, weights)