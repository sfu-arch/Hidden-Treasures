from Autodiff import *
with Graph() as g:
  x = Variable(1., name='x')
  y = Variable(2., name='y')
  z = Variable(3., name='z')
  loss = x*y+z
  
  ordering = topological_sort(loss, g)
  my_graph = make_graph(ordering)
  
import pygraphviz as pgv
from graphviz2drawio import graphviz2drawio
import io
graph_str = "{}".format(my_graph)
xml = graphviz2drawio.convert(graph_str)
with open('1.xml', 'w') as f:
  f.write(xml)
