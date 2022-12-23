from graphviz import Digraph

def make_graph(graph):
    """Allows us to visualize the computation graph directly in a Jupyter notebook.
    must have graphviz module installed. Takes as input the topological sorted ordering
    after calling the Session class"""
    f = Digraph()
    f.attr(rankdir='LR', size='10, 8')
    f.attr('node', shape='circle')
    for node in graph:
        shape = 'box' if isinstance(node, Placeholder) else 'circle'
        f.node(node.name, label=node.name.split('/')[0], shape=shape)
    for node in graph:
        if isinstance(node, Operator):
            for e in node.inputs:
                f.edge(e.name, node.name, label=e.name)
    return f