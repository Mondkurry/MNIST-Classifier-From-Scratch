import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MLUtils import loadingAnimation, printPurple, plot_function, displayPower
import random
from graphviz import Digraph, render

# * Value class

# class Value:
#     """ stores a single scalar value and its gradient """

#     def __init__(self, data, _children=(), _op=''):
#         self.data = data
#         self.grad = 0
#         # internal variables used for autograd graph construction
#         self._backward = lambda: None
#         self._prev = set(_children)
#         self._op = _op # the op that produced this node, for graphviz / debugging / etc

#     def __add__(self, other):
#         other = other if isinstance(other, Value) else Value(other)
#         out = Value(self.data + other.data, (self, other), '+')

#         def _backward():
#             self.grad += out.grad
#             other.grad += out.grad
#         out._backward = _backward

#         return out

#     def __mul__(self, other):
#         other = other if isinstance(other, Value) else Value(other)
#         out = Value(self.data * other.data, (self, other), '*')

#         def _backward():
#             self.grad += other.data * out.grad
#             other.grad += self.data * out.grad
#         out._backward = _backward

#         return out

#     def __pow__(self, other):
#         assert isinstance(other, (int, float)), "only supporting int/float powers for now"
#         out = Value(self.data**other, (self,), f'**{other}')

#         def _backward():
#             self.grad += (other * self.data**(other-1)) * out.grad
#         out._backward = _backward

#         return out

#     def relu(self):
#         out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

#         def _backward():
#             self.grad += (out.data > 0) * out.grad
#         out._backward = _backward

#         return out

#     def backward(self):

#         # topological order all of the children in the graph
#         topo = []
#         visited = set()
#         def build_topo(v):
#             if v not in visited:
#                 visited.add(v)
#                 for child in v._prev:
#                     build_topo(child)
#                 topo.append(v)
#         build_topo(self)

#         # go one variable at a time and apply the chain rule to get its gradient
#         self.grad = 1
#         for v in reversed(topo):
#             v._backward()

#     def __neg__(self): # -self
#         return self * -1

#     def __radd__(self, other): # other + self
#         return self + other

#     def __sub__(self, other): # self - other
#         return self + (-other)

#     def __rsub__(self, other): # other - self
#         return other + (-self)

#     def __rmul__(self, other): # other * self
#         return self * other

#     def __truediv__(self, other): # self / other
#         return self * other**-1

#     def __rtruediv__(self, other): # other / self
#         return other * self**-1

#     def __repr__(self):
#         return f"Value(data={self.data}, grad={self.grad})"

class Value:
    """ stores a value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
            
        out._backward = _backward

        return out
        

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            
            
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    def matmul(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(np.matmul(self.data , other.data), (self, other), 'matmul')
        def _backward():
            self.grad += np.dot(out.grad,other.data.T)
            other.grad += np.dot(self.data.T,out.grad)
            
            
        out._backward = _backward

        return out
    def softmax(self):

        out =  Value(np.exp(self.data) / np.sum(np.exp(self.data), axis=1)[:, None], (self,), 'softmax')
        softmax = out.data
        def _backward():
            self.grad += (out.grad - np.reshape(
            np.sum(out.grad * softmax, 1),
            [-1, 1]
              )) * softmax
        out._backward = _backward

        return out

    def log(self):
        out = Value(np.log(self.data),(self,),'log')
        def _backward():
            self.grad += out.grad/self.data
        out._backward = _backward

        return out
    
    
    def reduce_sum(self,axis = None):
        out = Value(np.sum(self.data,axis = axis), (self,), 'REDUCE_SUM')
        
        def _backward():
            output_shape = np.array(self.data.shape)
            output_shape[axis] = 1
            tile_scaling = self.data.shape // output_shape
            grad = np.reshape(out.grad, output_shape)
            self.grad += np.tile(grad, tile_scaling)
            
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            #print(v)
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    
class Visualizer():
    def __init__(self, root, rankdir = 'LR'):
        self.root = root
        self.nodes, self.edges = self.trace()
        self.format = 'pdf'
        self.rankdir = rankdir
    
    def trace(self):
            nodes, edges = set(), set()
            
            def build(current):
                if current not in nodes:
                    nodes.add(current)
                    for child in current._prev:
                        edges.add((child, current))
                        build(child)
            build(self.root)
            
            return nodes, edges
    
    def __call__(self, output = False):
        drawing = Digraph(format=self.format)
        drawing.attr(rankdir=self.rankdir)  # Set the rankdir attribute here
        
        for node in self.nodes:
            uid = str(id(node))
            label = "{%s | data %.4f | grad %.4f}" % (node.label, node.data, node.grad)
            drawing.node(name=uid, label=label, shape='record')
            
            if node._operation:
                drawing.node(name=uid + node._operation, label=node._operation)
                drawing.edge(uid + node._operation, uid)
                    
        for node1, node2 in self.edges:
            drawing.edge(str(id(node1)), str(id(node2)) + node2._operation)
            
        return drawing