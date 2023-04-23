# Simple-_Perceptron
This is a simple perceptron that I created completely by myself, without referring to other people's code, while looking at the slides of a machine learning workshop. It may or may not be correct.

This program can execute NN_2D and NN_3D by itself.


## APIs
~~~ Python
NN_2D(file_name, itr=11, weight=[-0.5, 0.7, 0.3], param=[0, 0], alpha=0.01)
~~~
Used when there are two explanatory variables
- file_name : input file name(requirement)
- itr : iteration
- weight : initial weight
- param : please ignore this param
- alpha : renewal rate

~~~ Python
NN_3D(file_name, itr=20, weight=[-0.5, 0.7, 0.7, 0.1], param=[0, 0], alpha=0.01)
~~~
Used when there are three explanatory variables
- file_name : input file name(requirement)
- itr : iteration
- weight : initial weight
- param : please ignore this param
- alpha : renewal rate
