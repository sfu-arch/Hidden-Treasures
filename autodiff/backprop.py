from fundamental_operators import *
##############################
#####  Topological sort  #####
##############################
def topological_sort(head_node=None, graph=_g):
    """Performs topological sort of all nodes prior to and 
    including the head_node. 
    Args:
        graph: the computational graph. This is the global value by default
        head_node: last node in the forward pass. The "result" of the graph.
    Returns:
        a sorted array of graph nodes.
    """
    vis = set()
    ordering = []
    
    def _dfs(node):
        if node not in vis:
            vis.add(node)
            if isinstance(node, Operator):
                for input_node in node.inputs:
                    _dfs(input_node)
            ordering.append(node)
            
    if head_node is None:
        for node in graph.operators:
            _dfs(node)
    else:
        _dfs(head_node)
        
    return ordering
    
  
##############################
#####    Forward pass    #####
##############################    
def forward_pass(order, feed_dict={}):
    """ Performs the forward pass, returning the output of the graph.
    Args:
        order: a topologically sorted array of nodes
        feed_dict: a dictionary values for placeholders.
    Returns:
        1. the final result of the forward pass.
        2. directly edits the graph to fill in its current values.
    """
    for node in order:
        
        if isinstance(node, Placeholder):
            node.value = feed_dict[node.name]
                    
        elif isinstance(node, Operator):
            node.value = node.forward(*[prev_node.value for prev_node in node.inputs])

    return order[-1].value
    
##############################
#####    Backward pass   #####
##############################  
def backward_pass(order, target_node=None):
    """ Perform the backward pass to retrieve gradients.
    Args:
        order: a topologically sorted array of graph nodes.
               by default, this assigns the graident of the final node to 1
    Returns:
        gradients of nodes as listed in same order as input argument
    """
    vis = set()
    order[-1].gradient = 1
    for node in reversed(order):
        if isinstance(node, Operator):
            inputs = node.inputs
            grads = node.backward(*[x.value for x in inputs], dout=node.gradient)
            for inp, grad in zip(inputs, grads):
                if inp not in vis:
                    inp.gradient = grad
                else:
                    inp.gradient += grad
                vis.add(inp)
    return [node.gradient for node in order]