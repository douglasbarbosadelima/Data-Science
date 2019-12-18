import numpy as np
import math
from functools import reduce

class MKMeans:
    def __init__(self, k, max_rounds = 20, max_steps = 10):
        self.max_rounds_ = max_rounds
        self.max_steps = max_steps
        self.k = k
            
        
    def __predict(self, xs, C):
        return [np.argmin([*map(lambda c: np.inner(x-c, x-c), C)]) for x in xs]
    
    def best(self, x, C):
        return self.__predict([x], C)[0]
    
    def predict(self, xs):
        return self.__predict(xs, self.cluster_centers_)
     
    def fit(self, X):
        self.X = X
        h = ncompose(self.bestFit, self.max_rounds_)
        self.cluster_centers_ = h((math.inf, None))[1]
        
    def bestFit(self, pair):
        min_error, best_C = pair
        (sse, C) = self.afit()
        return (sse, C) if sse < min_error else pair 
        
    
    def random(self):
        n = len(self.X)
        return [self.X[i] for i in np.random.choice(n, self.k)]
    
    def afit(self):
        def centroids(pi):
            return [sum([y for y in Y])/len(Y) for Y in pi]
        
        def optJc(C):
            return {i: self.best(x, C) for i, x in enumerate(self.X) }
        
        def optJp(P):
            n = len(self.X)
            pi = [[self.X[i] for i in range(n) if P[i] == j] for j in range(self.k)]
            return centroids(pi) if all(pi) else self.random()
        
        h = ncompose(compose(optJp, optJc), self.max_steps)
        C = h(self.random())
        return (sq_error(self.X, C, self.__predict(self.X, C)), C)
 
def sq_error(X, Y, fn):
    return sum([np.inner(X[i] - Y[fn[i]], X[i]-Y[fn[i]]) 
                for i in range(len(X))])
    
def compose(f, g):
    return lambda x: f(g(x))

def ncompose(f, n):
    return reduce(compose, [f] * n, lambda x: x)
