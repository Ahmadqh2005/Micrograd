from .engine import Value

import random as random 
random.seed(42)

class Neuron :

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self,x):
        act = sum ( (xi * wi for xi, wi in zip(x, self.w)), self.b)
        out = act.tanh()
        return out 
    
    def parameters(self):
        return self.w +[self.b]
    
class Layer :
    def __init__(self, nin, nop):
        self.neurons = [Neuron(nin) for _ in range(nop)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters() ]
    

class MLP :
    def __init__(self, nin, nouts): # nin > number of input into neural_network # nots> list  of the size layers 
        
        sz = [nin] + nouts 
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x ):
        # we need to  pass the iput to layers 
        for layer in self.layers :
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
