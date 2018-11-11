This content is very helpful seeing from *https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/03B_Layers_API.ipynb*

The entire purpose of TensorFlow is to have a so-called computational graph that can be executed much more efficiently than if the same calculations were to be performed directly in Python. TensorFlow can be more efficient than NumPy because TensorFlow knows the entire computation graph that must be executed, while NumPy only knows the computation of a single mathematical operation at a time.

TensorFlow can also automatically calculate the gradients that are needed to optimize the variables of the graph so as to make the model perform better. This is because the graph is a combination of simple mathematical expressions so the gradient of the entire graph can be calculated using the chain-rule for derivatives.

TensorFlow can also take advantage of multi-core CPUs as well as GPUs - and Google has even built special chips just for TensorFlow which are called TPUs (Tensor Processing Units) and are even faster than GPUs.

A TensorFlow graph consists of the following parts which will be detailed below:
 - Placeholder variables used for inputting data to the graph.
 - Variables that are going to be optimized so as to make the convolutional network perform better.
 - The mathematical formulas for the convolutional neural network.
 - A so-called cost-measure or loss-function that can be used to guide the optimization of the variables.
 - An optimization method which updates the variables.
