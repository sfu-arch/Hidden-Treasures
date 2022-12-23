from nodes import *
############################################
##### Fundamental Operator Definitions #####
############################################
class add(Operator):
    count = 0
    """Binary addition operation."""
    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs=[a, b]
        self.name = f'add/{add.count}' if name is None else name
        add.count += 1
        
    def forward(self, a, b):
        return a+b
    
    def backward(self, a, b, dout):
        return dout, dout

class multiply(Operator):
    count = 0
    """Binary multiplication operation."""
    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs=[a, b]
        self.name = f'mul/{multiply.count}' if name is None else name
        multiply.count += 1
        
    def forward(self, a, b):
        return a*b
    
    def backward(self, a, b, dout):
        return dout*b, dout*a
    
class divide(Operator):
    count = 0
    """Binary division operation."""
    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs=[a, b]
        self.name = f'div/{divide.count}' if name is None else name
        divide.count += 1
   
    def forward(self, a, b):
        return a/b
    
    def backward(self, a, b, dout):
        return dout/b, dout*a/np.power(b, 2)
    
    
class power(Operator):
    count = 0
    """Binary exponentiation operation."""
    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs=[a, b]
        self.name = f'pow/{power.count}' if name is None else name
        power.count += 1
   
    def forward(self, a, b):
        return np.power(a, b)
    
    def backward(self, a, b, dout):
        return dout*b*np.power(a, (b-1)), dout*np.log(a)*np.power(a, b)
    
class matmul(Operator):
    count = 0
    """Binary multiplication operation."""
    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs=[a, b]
        self.name = f'matmul/{matmul.count}' if name is None else name
        matmul.count += 1
        
    def forward(self, a, b):
        return a@b
    
    def backward(self, a, b, dout):
        return dout@b.T, a.T@dout
    
    
############################################
#####       Operator overloading       #####
############################################
def node_wrapper(func, self, other):
    """ Check to make sure that the two things we're comparing are
    actually graph nodes. Also, if we use a constant, automatically
    make a Constant node for it"""
    if isinstance(other, Node):
        return func(self, other)
    if isinstance(other, float) or isinstance(other, int):
        return func(self, Constant(other))
    raise TypeError("Incompatible types.")

Node.__add__ = lambda self, other: node_wrapper(add, self, other)
Node.__mul__ = lambda self, other: node_wrapper(multiply, self, other)
Node.__div__ = lambda self, other: node_wrapper(divide, self, other)
Node.__neg__ = lambda self: node_wrapper(multiply, self, Constant(-1))
Node.__pow__ = lambda self, other: node_wrapper(power, self, other)
Node.__matmul__ = lambda self, other: node_wrapper(matmul, self, other)