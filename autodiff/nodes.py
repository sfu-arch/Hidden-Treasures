### This won't do anything other than allow us to check 
### if in object is a Graph node or not
from graph import *


class Node:
    def __init__(self,g=None):
        self._g = g
        pass
      
 ### Placeholders ###
class Placeholder(Node):
    """An placeholder node in the computational graph. This holds
    a node, and awaits further input at computation time.
    Args: 
        name: defaults to "Plc/"+count
        dtype: the type that the node holds, float, int, etc.
    """
    count = 0
    def __init__(self, name,g,dtype=float,):
        Node.__init__(self, g)
        self._g.placeholders.add(self)
        self.value = None
        self.gradient = None
        self.name = f"Plc/{Placeholder.count}" if name is None else name
        Placeholder.count += 1
        
    def __repr__(self):
        return f"Placeholder: name:{self.name}, value:{self.value}"
        
### Constants ###      
class Constant(Node):
    """A constant node in the computational graph.
    Args: 
        name: defaults to "const/"+count
        value: a property protected value that prevents user 
               from reassigning value
    """
    count = 0
    def __init__(self, value, g, name=None):
        Node.__init__(self, g)
        self._g.constants.add(self)
        self._value = value
        self.gradient = None
        self.name = f"Const/{Constant.count}" if name is None else name
        Constant.count += 1
        
    def __repr__(self):
        return f"Constant: name:{self.name}, value:{self.value}"
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self):
        raise ValueError("Cannot reassign constant")
        self.value = None
        self.gradient = None
        self.name = f"Plc/{Placeholder.count}" if name is None else name
        Placeholder.count += 1
        
    def __repr__(self):
        return f"Placeholder: name:{self.name}, value:{self.value}"
      
### Variables ###
class Variable(Node):
    """An variable node in the computational graph. Variables are
    automatically tracked during graph computation.
    Args: 
        name: defaults to "var/"+count
        value: a mutable value
    """
    count = 0
    def __init__(self, value, name=None, g = None):
        Node.__init__(self, g)
        self._g.variables.add(self)
#        _g.variables.add(self)
        self.value = value
        self.gradient = None
        self.name = f"Var/{Variable.count}" if name is None else name
        Variable.count += 1
        
    def __repr__(self):
        return f"Variable: name:{self.name}, value:{self.value}"
      
### Operators ###
class Operator(Node):
    """An operator node in the computational graph.
    Args: 
        name: defaults to "operator name/"+count
    """
    def __init__(self, name='Operator', g = None):
        Node.__init__(self, g)
        self._g.operators.add(self)
        self.value = None
        self.inputs = []
        self.gradient = None
        self.name = name
    
    def __repr__(self):
        return f"Operator: name:{self.name}"