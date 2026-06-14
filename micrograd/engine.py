import math as math

class Value :
    def __init__(self, data, _children=(), _op="", label=""):
        self.data= data
        self.grad= 0.0
        self.prev= _children
        self._backward= lambda : None
        self._op = _op
        self.label= label 

    def __repr__ (self):
        return f"data = {self.data}  "
    
    def __add__(self, other):
        other= other if isinstance(other, Value) else Value(other)
        out= Value(self.data + other.data, (self, other), "+" )
        
        def _backward():
            self.grad +=1* out.grad
            other.grad +=1* out.grad
        out._backward = _backward
        
        return out 
    def __neg__(self):  # -self 
        return self * -1
    
    def __sub__(self, other)  : # self - other 
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value( other)
        out = Value( self.data * other.data , (self, other), "*" )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other): # other for now is just a (int or float)
        assert isinstance (other, (int, float))
        out = Value(self.data **other , (self, ), f"**{other}")

        def _backward():
            self.grad += out.grad *(other * (self.data**(other - 1)))
        out._backward =  _backward
        return out 

    def __truediv__(self, other) : # self / other  
        return self * other**-1
    
    def __rtruediv__(self, other):  # other / self
        return Value(other) * self**-1
    
    def __rsub__(self, other):  # other - self
        return Value(other) + (-self)

    def tanh(self) :
        x =self.data 
        t = ( math.exp( 2 * x) -1 ) / ( math.exp( 2 * x) +1 )
        out= Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1- t**2)* out.grad 
        out._backward= _backward
        return out
    
    def exp(self):
        x= self.data
        out= Value(math.exp(x), (self, ), "exp")

        def _backward():
            self.grad += out.grad * out.data 
        out._backward= _backward 
        return out 

    def __rmul__(self, other):  # other * self 
        return self.__mul__(other)
    
    def __radd__(self, other): # other + self
        return self + other
    
    def backward(self):
        # 1. Create an empty list to store the nodes in order
        topo = []
        # 2. Keep track of nodes we have already seen so we don't count them twice
        visited = set()
        
        # 3. A helper function to build the graph from left to right
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                # Before adding myself to the list, I must check all my previous children!
                for child in v.prev:
                    build_topo(child)
                # Once all children are processed, add myself to the list
                topo.append(v)

        # Start the sorting process from the output node (self)
        build_topo(self)
        
        # 4. Set the gradient of the final output node to 1.0
        self.grad = 1.0
        
        # 5. Go through the list backwards (from output to inputs)
        for node in reversed(topo):
            # print(node)
            node._backward()

