import collections
import numpy as np
import random
import torch

# Replay buffer
class ReplayBuffer:
    
    # create replay buffer of size N
    def __init__(self, N):
        self.buf = collections.deque(maxlen = N)
    
    # add: add a transition (s, a, r, s2, d)
    def add(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
    
    # sample: return minibatch of size n
    def sample(self, n, t):
        minibatch = random.sample(self.buf, n)
        S, A, R, S2, D = [], [], [], [], []
        
        for mb in minibatch:
            s, a, r, s2, d = mb
            S += [s]; A += [a]; R += [r]; S2 += [s2]; D += [d]

        return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D)