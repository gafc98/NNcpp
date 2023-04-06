import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
import matplotlib.pyplot as plt
import pickle

class NN:

    def __init__(self, hls = [100], n_input = 1, n_out = 1, lr = 0.1):
        self.set_lrs(hls, n_input, n_out)
        self.n_input = n_input
        self.n_output = n_out
        self.lr = lr

    def set_lrs(self, hls = [100], n_input = 1, n_out = 1):
        self.lrs = [n_input]
        for l in hls:
            self.lrs.append(l)
        self.lrs.append(n_out)
        return self

    def eval(self, x, p = None):
        if self and not p:
            p = self.p

        x = x.reshape(-1, 1)
        # hidden layers
        for W, b in p:
            x = np.tanh(np.dot(W, x) + b)
            #print(x)

        return x

    def set_init_params(self):
        set_weights = lambda x, low, up: (up - low) * (x - 0.5) + (up + low) * 0.5
        low = -0.001
        high = 0.001
        W = set_weights(np.random.rand(self.lrs[1], self.lrs[0]), -1, 1)
        b = set_weights(np.random.rand(self.lrs[1],1), low, high)
        params = [(W, b)]
        for i in range(1, len(self.lrs) - 1):
            W = set_weights(np.random.rand(self.lrs[i+1], self.lrs[i]), -1, 1)
            b = set_weights(np.random.rand(self.lrs[i+1],1), -1, 1)
            params.append( (W, b) )
        self.p = params
        return self

    def get_grad(self, x):
        # compute activations
        x = x.reshape(-1, 1)
        # hidden layers
        o = [x]
        i = 0
        for W, b in self.p:
            #print(W)
            o.append(np.tanh(np.dot(W, o[i]) + b))
            i =+ 1
        o.pop(0)

        for act in o:
            print(act)
            print()

        # compute deltas
        deltas = [(np.ones(self.p[-1][1].size), None)]
        i = 0
        for l in range(len(self.p)-1, 0, -1):
            #print(deltas[i])
            #print(self.p[l][0])
            #print(self.p[l][1])
            print(l)
            print(deltas[i][0])
            print(self.p[l][0])
            delta_w = np.dot(deltas[i][0], self.p[l][0]) * 2 * o[l] * (1 - o[l])
            print(delta_w)
            delta_b = np.dot(deltas[i][0], self.p[l][1]) * 1
            deltas.append((delta_w, delta_b))
            i += 1

        deltas.reverse()

        print("deltas")
        for delta in deltas:
            print(delta)
            print()


        grad = []
        print()
        for i in range(len(o)):
            print(i)
            grad.append((deltas[i][0] * o[i][0], deltas[i][1]))

        for g in grad:
            print(g)
            print()







nn = NN([2, 2]).set_init_params()
print("params")
for pl in nn.p:
    print(pl)
    print()

#print(type(nn.p[2][1]))
#print([np.ones(nn.p[-1][1].size)])
#print()
#print(nn.p)
x_eval = np.ones(1)
nn.get_grad(x_eval)

g = grad(lambda p, x: nn.eval(x, p))

grad_g = g(nn.p, x_eval)
print("grad_g")
for g in grad_g:
    print(g)
    print()