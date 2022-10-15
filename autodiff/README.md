# Dependencies

**graphviz**
```
linux:sudo apt-get install graphviz
OSX: brew install graphviz
```
**pygraphviz**
```bash
OSX: pip install --global-option=build_ext --global-option="-I/opt/homebrew/Cellar/graphviz/3.0.0/include/" --global-option="-L/opt/homebrew/Cellar/graphviz/3.0.0/lib" pygraphviz

Specify path to graphviz libraries
linux: pip install --global-option=build_ext --global-option="-I/usr/local/include/" --global-option="-L/usr/local/lib" pygraphviz
```

grpahviz2drawio (for importing into lucidchart)
```
pip3 install graphviz2drawio
```

```
python3 test.py
```


# Autodiff python library


Derived from here: https://towardsdatascience.com/build-your-own-automatic-differentiation-program-6ecd585eec2a

