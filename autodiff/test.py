from Autodiff import *
with Graph() as g:
  x = Variable(1., name='x')
  y = Variable(2., name='y')
  z = Variable(3., name='z')
  loss = x*y+z
  
  ordering = topological_sort(loss, g)
  my_graph = make_graph(ordering)
  print(my_graph)