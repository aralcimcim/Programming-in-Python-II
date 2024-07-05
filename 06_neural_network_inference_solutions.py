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

Example solutions for tasks in the provided tasks file.
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

class ImageNN(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, n_units_per_layer: list):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        for n_units in n_units_per_layer:
            layers.append(nn.Linear(in_features=in_features, out_features=n_units))
            layers.append(nn.ReLU())
            in_features = n_units
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features=in_features, out_features=out_features)
    
    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.layers(x)
        return self.output_layer(x)


image_nn = ImageNN(in_features=3 * 100 * 100, out_features=10, n_units_per_layer=[4096, 2048, 2048, 1024])
print(image_nn)
print("\nApplying ImageNN")
print(f"input tensor shape: {input_tensor.shape}")
output_tensor = image_nn(input_tensor)
print(f"output tensor shape: {output_tensor.shape}")


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

class FeatureExtractor(nn.Module):
    
    def __init__(self, n_dims: int, reduction: str = "max"):
        super().__init__()
        self.n_dims = n_dims
        if reduction == "max":
            # torch.max returns a named tuple, so wrap it within a lamda that
            # extracts the actual maximum values
            self.f = lambda x, dim: torch.max(x, dim=dim).values
        elif reduction == "mean":
            self.f = torch.mean
        else:
            raise ValueError(f"'reduction' must bei either 'max' or 'mean', not '{reduction}'")
    
    def forward(self, x: torch.Tensor):
        # This implementation simply invokes the reduction function n_dims times
        # and always applies it on the last dimension (dim=-1). This makes it
        # very easy to extend it to different reduction functions, however, it
        # does not take advantage in case there are more efficient invocation
        # calls such as immediately specifying all dimensions that need to be
        # reduced in the function torch.mean.
        for _ in range(self.n_dims):
            x = self.f(x, dim=-1)
        return x


feat = FeatureExtractor(n_dims=2, reduction="max")
print("\nApplying FeatureExtractor")
print(f"input tensor shape: {input_tensor.shape}")
output_tensor = feat(input_tensor)
print(f"output tensor shape: {output_tensor.shape}")


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

class CNN(nn.Module):
    
    def __init__(self, n_input_channels: int, n_conv_layers: int, n_kernels: int, kernel_size: int):
        """CNN, consisting of ``n_hidden_layers`` linear layers, using relu
        activation function in the hidden CNN layers.
        
        Parameters
        ----------
        n_input_channels: int
            Number of features channels in input tensor
        n_conv_layers: int
            Number of conv. layers
        n_kernels: int
            Number of kernels in each layer
        kernel_size: int
            Number of features in output tensor
        """
        super().__init__()
        
        layers = []
        n_concat_channels = n_input_channels
        for i in range(n_conv_layers):
            # Add a CNN layer with appropriate padding to keep the dimensions the same
            layer = nn.Conv2d(
                in_channels=n_concat_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            layers.append(layer)
            self.add_module(f"conv_{i:0{len(str(n_conv_layers))}d}", layer)
            # Prepare for concatenated input
            n_concat_channels = n_kernels + n_input_channels
            n_input_channels = n_kernels
        
        self.layers = layers
    
    def forward(self, x: torch.Tensor):
        """Apply CNN to ``x``.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(n_samples, n_input_channels, height, width)``
        
        Returns
        ----------
        torch.Tensor
            Output tensor of shape ``(n_samples, n_output_channels, h', w')``
        """
        skip_connection = None
        output = None
        
        # Apply layers module
        for layer in self.layers:
            # If previous output and skip_connection exist, concatenate them and
            # store previous output as new skip_connection. Otherwise, use x as
            # input and store it as skip_connection.
            if skip_connection is not None:
                assert output is not None
                inp = torch.cat([output, skip_connection], dim=1)
                skip_connection = output
            else:
                inp = x
                skip_connection = x
            # Apply CNN layer
            output = torch.relu(layer(inp))
        
        return output


cnn = CNN(n_input_channels=3, n_conv_layers=16, n_kernels=32, kernel_size=3)
print("\nApplying CNN")
print(f"input tensor shape: {input_tensor.shape}")
output_tensor = cnn(input_tensor)
print(f"output tensor shape: {output_tensor.shape}")
