# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

Tasks for self-study. Try to solve these tasks on your own and compare your
solutions to the provided solutions file.
"""

import torch
import torch.nn as nn

# Input minibatch with 4 samples: 100 by 100 images and 3 color channels
input_tensor = torch.arange(4 * 3 * 100 * 100, dtype=torch.float32).reshape((4, 3, 100, 100))


#
# Task 1
#

# Implement a class ImageNN as PyTorch module that works as follows: The input
# is expected to be an image of shape (C, H, W) that needs to be flattened and
# then forwarded to arbitrarily deep fully-connected layers. The output layer
# will ultimately transform the last layer output to a specified number of
# output features. As non-linear activation functions, ReLU should be chosen.
# The __init__ method should take the following arguments:
#   > in_features: The number of (flattened) input features which will be the
#     input to the first (hidden) layer (see "n_units_per_layer").
#   > out_features: The number of output features of the last layer. After the
#     output layer, no activation function must follow.
#   > n_units_per_layer: A list of integers that specify the number of units per
#     (hidden) layer, i.e., each integer represents the number of output
#     features of this particular layer.

# Your code here #


#
# Task 2
#

# Implement a class FeatureExtractor as PyTorch module which applies a
# specifiable reduction to the last n dimensions of an arbitrary input tensor.
# The __init__ method should take the following arguments:
#   > n_dims: The amount of dimensions at the end of an input vector that should
#     be reduced to a single value via a reduction function.
#   > reduction: A string specifying the reduction function that should be
#     applied. Can either be "max" to extract the maximum value or "mean" to
#     calculate the average value over the dimensions that need to be reduced.
# Example:
#   > input tensor shape: (A, B, C, D, E)
#   > n_dims = 3, reduction = "max"
#   > output tensor shape: (A, B), where the last n_dims = 3 dimensions were
#     reduced to their respective maximums

# Your code here #


#
# Task 3
#

# Implement a CNN as PyTorch module which applies convolutional layers with an
# activation function of your choice. The CNN should furthermore employ
# skip-connections between the convolutional layers. The skip-connection tensors
# should be concatenated (e.g., "DenseNet") instead of using an element-wise sum
# (e.g., "ResNet"). This can be done by concatenating the output channels of the
# current layer with the output of the layer below and feeding this concatenated
# tensor into the next layer. The __init__ method should take the following
# arguments:
#   > n_conv_layers: The number of conv. layers in the network.
#   > n_kernels: The number of kernels in each conv. layer.
#   > kernel_size: The size of the kernels in the conv. layers; you can
#     expect this to be an odd integer.
#   > n_input_channels: The number of input channels.
#
# Notes:
# You will need to apply padding at the borders of the CNN, otherwise you will
# not be able to concatenate the layer outputs. nn.Sequential will not work
# here, you will need to store a list of layers and iterate over it in the
# "forward" method to perform the skip-connections at each iteration. Do not
# forget to register each layer using "self.add_module()". The exact design of
# the CNN is your choice.

# Your code here #
