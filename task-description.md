initalize a jj repo and each of the following tasks should be at least one commit:
1. take the heat "neural ode.jl" file and make a normal julia file out of it. Split the manifest and toml into seperate files
2. instantiate the julia environment
3. Make an function that trains a model on the heat equation problem in the file where the model can be chosen as input, also create functions to:
   1. run the heat equation multiple times and collect the data from the multiple runs as training data for the model
   2. a function to evaluate the performance of the model using root mean squared error
   3. an ablility to save and load the models
4. try out two different approaches
   1. one with a neural ode like in the example code
   2.  one with a neural network that gets an input, learns some parameters and puts them into a then a neural ode and then another network that takes the neural ode output and puts it into another neural network that then retrieves the parameters from the result
   3.  a covolutional neural network
5.  save all three models
6.  include a way to run the models from the saved file
7.  plot the result
