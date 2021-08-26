#Reasonable ideas to increase policy in ddpg

Ideas from: https://github.com/yanpanlau/DDPG-Keras-Torcs/issues/11


## use rgb normalized images instead of grayscale
using 3 channels and bringing them between 0.0 and 1.0

## use 4 consecutive images as state
this would lead to a (64,64,12) input_shape then

## use batch normalization
someone mentioned to do it after each conv layer
