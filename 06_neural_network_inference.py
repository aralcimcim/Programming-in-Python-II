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

In this file, we will learn how to use PyTorch to create and use our neural
network (NN) layers and complete NNs.
"""

################################################################################
# PyTorch torch.nn.Module - Creating a custom neural network layer
################################################################################

# The base class for all neural network modules is the torch.nn.Module class.
# All your custom networks/layers/modules should be derived from this class. It
# is possible to nest PyTorch modules, i.e., you can use other modules within a
# module, e.g., you can use linear layer modules within your neural network
# module. PyTorch modules are peculiar, since the attributes can have special
# meanings.

import numpy as np
import torch
import torch.nn as nn

# On reproducibility in PyTorch:
# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)  # Set a known random seed for reproducibility


#
# Very simple PyTorch module
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
#

# Let's create a very simple PyTorch module "ProductModule". This module should
# take a PyTorch tensor as input and multiply it with a scalar "a". We have to
# create a custom __init__ method to receive our scalar "a" and a "forward"
# method. The __init__ method essentially sets up the module's architecture. The
# "forward" method is the method that will be called if our module is applied to
# input tensors, i.e., when data is passed to our module to produce output (the
# "flow" through the architecture).

# Create a PyTorch module "ProductModule"
class ProductModule(nn.Module):
    
    def __init__(self, a: float):
        # Here, we define our __init__ method. In our case, the method takes one
        # argument, which is the scalar to multiply an input tensor with. Do not
        # forget to call the __init__ method of the parent module nn.Module
        super().__init__()
        # Convert the scalar "a" to a float32 tensor and store it as attribute
        self.a = torch.tensor(a, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor):
        # Here, we define our "forward" method, which is called when the module
        # is applied to input tensors. In our case, our input tensor is "x", but
        # we could have multiple input parameters. Our simple module just does
        # a multiplication with our scalar "a".
        output = x * self.a
        return output


# Done! Now we can use our PyTorch module:
product_module = ProductModule(a=2)  # Create a module instance
input_tensor = torch.arange(5, dtype=torch.float32)  # Create some input tensor
print(f"input tensor: {input_tensor}")
# Apply our module. This will ultimately invoke our "forward" method, but make
# sure to always use this approach, since PyTorch modules can contain hooks that
# are automatically called (they are skipped if you call "forward" directly)
output_tensor = product_module(input_tensor)
print(f"output tensor: {output_tensor}")


#
# Adding trainable parameters to our module
# https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
#

# Let's create a PyTorch module that is a simple linear NN layer. A fully
# connected feed-forward linear layer computes the product of an input tensor
# with a weight matrix. In order to later train these weights such that our NN
# solves some task, we will specify it as trainable tensor (torch.nn.Parameter).

# Create a PyTorch module "LinearLayer"
class LinearLayer(nn.Module):
    
    def __init__(self, n_input_features: int, n_output_features: int):
        """Simple linear layer.
        
        Parameters
        ----------
        n_input_features: int
            Number of features in input tensor
        n_output_features: int
            Number of features in output tensor
        """
        super().__init__()
        # We can still use other Python objects as "normal" attributes
        self.n_output_features = n_output_features
        
        # In order to take an input tensor with n_input_features features and
        # produce an output tensor with n_output_features features, we require
        # a weight matrix of size (n_input_features, n_output_features), which
        # we will multiply with the input tensor.
        weight_tensor = torch.empty(size=(n_input_features, n_output_features), dtype=torch.float32)
        # We want to train this tensor, so we create a torch.nn.Parameter
        self.weights = nn.Parameter(data=weight_tensor, requires_grad=True)
        
        # self.weights is now a PyTorch parameter that can be optimized during
        # training (self.weights = weight_tensor would not be sufficient!)
        
        # Important: Specifying the PyTorch parameter as PyTorch module
        # attribute will automatically register the parameter with the module.
        # Assigning a different object to the module attribute self.weights
        # results in an error! Example:
        # self.weights = torch.tensor(4)  # This would raise an exception!
        
        # We initialize our weights with random values from a uniform
        # distribution. For this, we use the functions provided in torch.nn.init
        torch.nn.init.uniform_(self.weights)
    
    def forward(self, x: torch.Tensor):
        """Apply linear layer weight ``self.w`` to ``x``.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(n_input_features,)``
        
        Returns
        ----------
        torch.Tensor
            Output tensor of shape ``(n_output_features,)``
        """
        # Apply our weights by using a matrix multiplication of inputs and weights
        output = x.matmul(self.weights)
        return output


# We can now use our linear layer:
linear_layer = LinearLayer(n_input_features=5, n_output_features=2)
# We can access the trainable parameters using the provided "parameters" method
print("\nLinearLayer")
print(f"module parameters: {list(linear_layer.parameters())}")
input_tensor = torch.arange(5, dtype=torch.float32)
print(f"input tensor: {input_tensor}")
output_tensor = linear_layer(input_tensor)
print(f"output tensor: {output_tensor}")


################################################################################
# PyTorch torch.nn.Module - Predefined modules
################################################################################

# There are a lot of predefined modules available, including various NN layer
# types, regularization modules, etc., some of which we will see later in this
# document. Official documentation: https://pytorch.org/docs/stable/nn.html

# Our LinearLayer module is nice, but the PyTorch predefined module
# torch.nn.Linear is better optimized, includes an optional bias weight,
# typically performs suitable weight initialization (see docs) and performs
# proper broadcasting over multiple samples. In such cases, one would prefer the
# predefined version unless there is need for something specific that the
# predefined version does not include.
predefined_linear_layer = nn.Linear(in_features=5, out_features=2, bias=False)
print("\nnn.Linear")
print(f"module parameters: {list(predefined_linear_layer.parameters())}")
input_tensor = torch.arange(5, dtype=torch.float32)
print(f"input tensor: {input_tensor}")
output_tensor = predefined_linear_layer(input_tensor)
print(f"output tensor: {output_tensor}")


################################################################################
# PyTorch torch.nn.Module - Combining modules
################################################################################

# PyTorch modules can utilize other PyTorch modules. For example, we can build
# a fully-connected feed-forward neural network as a PyTorch module, which
# consists of multiple linear layers by using the torch.nn.Linear module.

# Create a PyTorch module "SNN" for a Self-Normalizing Neural Network ("SNN")
# https://papers.nips.cc/paper/2017/file/5d44ee6f2c3f71b73125876103c8f6c4-Paper.pdf
class SNN(nn.Module):
    
    def __init__(self, n_input_features: int, n_hidden_units: int, n_output_features: int):
        """Fully-connected feed-forward neural network, consisting of 5 linear
        layers, using selu activation function in the hidden layers.
        
        Parameters
        ----------
        n_input_features: int
            Number of features in input tensor
        n_hidden_units: int
            Number of units in each hidden layer
        n_output_features: int
            Number of features in output tensor
        """
        super().__init__()
        
        # We want to use 5 linear layers
        
        # First layer takes input tensor of shape (n_input_features,) and
        # creates output tensor of shape (n_hidden_units,)
        self.layer_0 = nn.Linear(in_features=n_input_features, out_features=n_hidden_units)
        # We need to make sure that the weights are initialized according to
        # selu theory (normal distribution with mean=0, std=1 / sqrt(n_incoming))
        torch.nn.init.normal_(self.layer_0.weight, 0, 1 / np.sqrt(self.layer_0.in_features))
        
        # Second layer takes input tensor of shape (n_hidden_units,) and
        # creates output tensor of shape (n_hidden_units,)
        self.layer_1 = nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units)
        torch.nn.init.normal_(self.layer_1.weight, 0, 1 / np.sqrt(self.layer_1.in_features))
        
        # Same for third layer
        self.layer_2 = nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units)
        torch.nn.init.normal_(self.layer_2.weight, 0, 1 / np.sqrt(self.layer_2.in_features))
        
        # Same for fourth layer
        self.layer_3 = nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units)
        torch.nn.init.normal_(self.layer_3.weight, 0, 1 / np.sqrt(self.layer_3.in_features))
        
        # The output layer takes input tensor of shape (n_hidden_units,) and
        # creates output tensor of shape (n_output_features,)
        self.layer_4 = nn.Linear(in_features=n_hidden_units, out_features=n_output_features)
        torch.nn.init.normal_(self.layer_4.weight, 0, 1 / np.sqrt(self.layer_4.in_features))
        
        # Important: Analogous to nn.Parameter, specifying a PyTorch module
        # (such as nn.Linear) as PyTorch module attribute will automatically
        # register the submodule with the module. Assigning a different object
        # to the module attribute, e.g., self.layer_0, results in an error!
        # self.layer_0 = torch.tensor(4)  # This would raise an exception!
    
    def forward(self, x: torch.Tensor):
        """Apply SNN to ``x``.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(n_samples, n_input_features)`` or
            ``(n_input_features,)``
        
        Returns
        ----------
        torch.Tensor
            Output tensor of shape ``(n_samples, n_output_features)`` or
            ``(n_output_features,)``
        """
        # Apply layer 0 and selu activation function
        x = self.layer_0(x)
        nn.functional.selu(x, inplace=True)
        # Short-cut for in-place: torch.nn.functional.selu_(x)
        
        # Apply layers 1-3 and (inplace) selu activation function
        x = self.layer_1(x)
        nn.functional.selu_(x)
        x = self.layer_2(x)
        nn.functional.selu_(x)
        x = self.layer_3(x)
        nn.functional.selu_(x)
        
        # Apply last layer (=output layer) without selu activation
        output = self.layer_4(x)
        return output


snn = SNN(n_input_features=5, n_hidden_units=16, n_output_features=2)
# The "parameters" method will return us all trainable parameters of the module,
# including the parameters of the submodules by default (recurse=True).
print("\nSNN")
print(f"module parameters: {list(snn.parameters(recurse=True))}")
input_tensor = torch.arange(5, dtype=torch.float32)
print(f"input tensor: {input_tensor}")
output_tensor = snn(input_tensor)
print(f"output tensor: {output_tensor}")

#
# Registering child modules manually
#

# If you do not want to add a child module as attribute directly, you can
# register it using the "self.add_module" method of the PyTorch module.


################################################################################
# PyTorch torch.nn.Module - Sending a PyTorch module to a device
################################################################################

# So far, our modules have been computed on the CPU, which is the default. Due
# to the PyTorch just-in-time compiler, this is already quite fast and optimized
# for our CPU. For larger NNs, especially CNNs, GPUs are much faster, as they
# are specialized to matrix operations (e.g., the matrix multiplication with the
# weight matrix). It would be nice if we could send our module, including all
# tensors/parameters in the computational graph, to our GPU and perform all
# computations there. Luckily, this is really easy using PyTorch. Naturally,
# this only works if you have a supported GPU.

# Specify device ("cpu" for CPU and "cuda:i" where "i" is the number of the GPU
# in your machine, or simply "cuda" to use the current CUDA device):
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the module, including tensors and computational graph to the GPU
snn.to(device=device)

# Our SNN instance is now on the GPU, and we perform computations on the GPU!
print("\nGPU NN" if torch.cuda.is_available() else "\nCUDA not available; regular CPU NN")
print(f"module parameters: {list(snn.parameters())}")
# Our input tensor needs to be on the GPU too:
input_tensor = torch.arange(5, dtype=torch.float32, device=device)
print(f"input tensor: {input_tensor}")
output_tensor = snn(input_tensor)
print(f"output tensor: {output_tensor}")

#
# Converting modules to other data types
#

# The convenience does not end here - we can even convert our module to a
# different data type easily! Assume we want to perform float64 computations on
# our CPU, then all we have to do is specify the new data type:
snn.to(device=torch.device("cpu"), dtype=torch.float64)

# Our SNN instance is now on the CPU, and we perform computations on the CPU
# using float64!
print("\nfloat64 NN")
print(f"module parameters: {list(snn.parameters())}")
# Our input tensor needs to be on the CPU in dtype float64 too:
input_tensor = torch.arange(5, dtype=torch.float64)
print(f"input tensor: {input_tensor}")
output_tensor = snn(input_tensor)
print(f"output tensor: {output_tensor}")

#
# Using float16
#

# Not all GPUs or CPUs support float16 computations. Modern GPUs typically do
# support float16 but sometimes with suboptimal optimization that only reduces
# GPU memory consumption but without other computational speed-ups. However,
# some special hardware comes with dedicated support for bfloat16 (for details,
# see https://cloud.google.com/tpu/docs/bfloat16).

# Let's try (regular) float16 on your machine! You can try bfloat16 as well.
try:
    snn.to(device=device, dtype=torch.float16)
    print("\nfloat16 NN")
    print(f"module parameters: {list(snn.parameters())}")
    input_tensor = torch.arange(5, device=device, dtype=torch.float16)
    print(f"input tensor: {input_tensor}")
    output_tensor = snn(input_tensor)
    print(f"output tensor: {output_tensor}")
except RuntimeError as e:
    print(f"\n{e}")

#
# Broadcasting over sample dimension
#

# Predefined PyTorch modules support broadcasting over a sample dimension, such
# that we can send a minibatch of samples into the module instead of computing
# the output one sample at a time.

print("\n4 samples in minibatch NN")
snn.to(device=device, dtype=torch.float32)  # Let's keep it at float32 here
# In torch.nn.Linear, the first dimension can be used as sample dimension. Our
# minibatch matrix for 4 samples would be of shape (4, 5):
input_tensor = torch.arange(4 * 5, device=device, dtype=torch.float32).reshape((4, 5))
print(f"input tensor shape: {input_tensor.shape}")
output_tensor = snn(input_tensor)
print(f"output tensor shape: {output_tensor.shape}")
# Broadcasting over the sample dimension was done automatically! Note: Always
# check the expected input dimensions for modules!


################################################################################
# Deep SNN example - torch.nn.Sequential
################################################################################

# For creating larger networks with more layers, we might want to automate the
# stacking of layers. This will also come in handy if you want to treat the
# number of layers as hyperparameter. torch.nn.Sequential allows us to create a
# PyTorch module by stacking arbitrary many PyTorch modules.

# Create a PyTorch module "DSNN" for a deep Self-Normalizing Neural Network
class DSNN(nn.Module):
    
    def __init__(self, n_input_features: int, n_hidden_layers: int, n_hidden_units: int, n_output_features: int):
        """Fully-connected feed-forward neural network, consisting of
        ``n_hidden_layers`` linear layers, using selu activation function in the
        hidden layers.
        
        Parameters
        ----------
        n_input_features: int
            Number of features in input tensor
        n_hidden_layers: int
            Number of hidden layers
        n_hidden_units: int
            Number of units in each hidden layer
        n_output_features: int
            Number of features in output tensor
        """
        super().__init__()
        
        # We want to use n_hidden_layers linear layers
        hidden_layers = []
        for _ in range(n_hidden_layers):
            # Add linear layer module to list of modules
            layer = nn.Linear(in_features=n_input_features, out_features=n_hidden_units)
            torch.nn.init.normal_(layer.weight, 0, 1 / np.sqrt(layer.in_features))
            hidden_layers.append(layer)
            # Add selu activation module to list of modules
            hidden_layers.append(nn.SELU())
            n_input_features = n_hidden_units
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # The output layer usually is separated to allow easy access to the
        # internal features (the model's data representation after the hidden
        # layers; see feature extraction example in 04_data_analysis.py)
        self.output_layer = nn.Linear(in_features=n_hidden_units, out_features=n_output_features)
        torch.nn.init.normal_(self.output_layer.weight, 0, 1 / np.sqrt(self.output_layer.in_features))
    
    def forward(self, x: torch.Tensor):
        """Apply deep SNN to ``x``.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(n_samples, n_input_features)`` or
            ``(n_input_features,)``
        
        Returns
        ----------
        torch.Tensor
            Output tensor of shape ``(n_samples, n_output_features)`` or
            ``(n_output_features,)``
        """
        # Apply hidden layers module
        hidden_features = self.hidden_layers(x)
        
        # Apply last layer (=output layer) without selu activation
        output = self.output_layer(hidden_features)
        return output


# According to the theory, SNN layers should normalize the activations between
# the layers to mean=0 and variance=1 automatically (if correctly initialized
# and receiving inputs with mean=0 and variance=1). In practice, even other
# inputs are normalized to some extent if the SNN is deep enough.
# Let's try it out:
dsnn = DSNN(n_input_features=32, n_hidden_layers=16, n_hidden_units=32, n_output_features=32)

# GPU will be much faster here
dsnn.to(device=device)
# Create some non-normalized input
input_tensor = torch.arange(32, dtype=torch.float32, device=device)
print("\nDeep SNN")
print(f"input tensor: {input_tensor}")
print(f"input tensor mean: {input_tensor.mean()}")
print(f"input tensor std: {input_tensor.std()}")
output_tensor = dsnn(input_tensor)
print(f"output tensor: {output_tensor}")
print(f"output tensor mean: {output_tensor.mean()}")
print(f"output tensor std: {output_tensor.std()}")
# The SNN output is indeed closer to mean=0 and variance=1 than the input


################################################################################
# RNN example
################################################################################

# PyTorch offers different RNN modules, such as LSTM or GRU, with specialized
# optimization for GPU/CUDA. However, for smaller RNNs, the computation on CPUs
# is typically faster (loops over sequences are slow on GPUs). The PyTorch LSTM
# implementation is furthermore rather restricted in its design.

# If you want to learn more about LSTMs and RNNs, please visit the RNN/LSTM
# course "LSTM and Recurrent Neural Nets" in the AI study or visit
# https://github.com/widmi/widis-lstm-tools.

# We will now program a very simple RNN which should process a sequence of input
# vectors. We will concatenate the input x from timestep t and the output of the
# hidden layer h at timestep t-1, and use these concatenated features as input
# for the hidden layer prediction at timestep t.
# In short: h_t = hidden_layer(x_t, h_{t-1})

# Create a PyTorch module "RNN" for a simple RNN
class RNN(nn.Module):
    
    def __init__(self, n_input_features: int, n_hidden_units: int, n_output_features: int):
        """Simple RNN consisting of one recurrent fully-connected layer with
        sigmoid activation function, followed by one fully-connected
        feed-forward output layer.
        
        Parameters
        ----------
        n_input_features: int
            Number of features in input tensor
        n_hidden_units: int
            Number of units in the hidden layer
        n_output_features: int
            Number of features in output tensor
        """
        super().__init__()
        # Create a fully-connected layer that expects the concatenated forward
        # features n_input_features and recurrent features n_hidden_units
        self.hidden_layer = nn.Linear(in_features=n_input_features + n_hidden_units, out_features=n_hidden_units)
        # Fully-connected output layer
        self.output_layer = nn.Linear(in_features=n_hidden_units, out_features=n_output_features)
        # We need some initial value for h_{t-1} at t=0. We will use a 0-vector
        self.h_init = torch.zeros(size=(n_hidden_units,), dtype=torch.float32)
    
    def forward(self, x: torch.Tensor):
        """Apply RNN to ``x``.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(n_sequence_positions, n_input_features)``
        
        Returns
        ----------
        torch.Tensor
            Output tensor of shape ``(n_output_features,)``
        """
        # Get initial h_{t-1} for t = 0
        h = self.h_init
        
        # Iterate over the sequence positions
        for x_t in x:
            # Concatenate x_t and h_{t-1}
            inp_t = torch.cat([x_t, h])
            # Compute new h from x_t and h_{t-1}
            h = self.hidden_layer(inp_t)
            # tanh non-linearity as activation function
            h = torch.tanh(h)
        
        # Last layer only sees h from last timestep
        output = self.output_layer(h)
        return output


# Create an instance of our RNN
rnn = RNN(n_input_features=3, n_hidden_units=16, n_output_features=2)
# CPU will be faster here (small matrices and loop)
# rnn.to(device="cpu")  # CPU is the default anyway
# Create some input sequence with length 8 and 3 features per position
input_tensor = torch.arange(8 * 3, dtype=torch.float32).reshape((8, 3))
print("\nRNN")
print(f"input tensor shape: {input_tensor.shape}")
output_tensor = rnn(input_tensor)
print(f"output tensor shape: {output_tensor.shape}")


################################################################################
# CNN example
################################################################################

# In convolutional neural networks (CNNs), weight kernels are slid (=convolved)
# along an input tensor. For 1D CNNs, the kernels are convolved over one
# dimension (e.g., the time dimension in a time series). For 2D CNNs, the
# kernels are convolved over 2 dimensions (e.g., the spatial dimensions in an
# image). For images as input tensors, this means that the hidden feature vector
# is actually a matrix since the spatial dimensions are not removed. This matrix
# will have smaller spatial dimensions than the input tensor due to the
# convolution (actually: cross-correlation) unless it is padded on the sides
# beforehand. Typically, odd values are used as kernel width and height to
# make padding symmetrical by adding "kernel_size // 2" values at both borders
# of the input image. Padding with 0-values is a common choice.

# Typically, hyperparameters that are optimized are the size of the weight
# kernels, the number of kernels, the stride, which is the number of
# positions the kernel is moved (step size), and the number of CNN layers.

# CNNs often employ pooling functions, which reduce a number of inputs to a
# scalar value. For example, one can apply max-pooling with a certain window
# size to replace a window of (n_pixels, n_pixels) to (1, 1), which decreases
# the size of the hidden feature matrices by taking only the maximum value.

# Create a PyTorch module "CNN" for a CNN
class CNN(nn.Module):
    
    def __init__(self, n_input_channels: int, n_hidden_layers: int, n_hidden_kernels: int, n_output_channels: int):
        """CNN, consisting of ``n_hidden_layers`` linear layers, using relu
        activation function in the hidden CNN layers.
        
        Parameters
        ----------
        n_input_channels: int
            Number of channels in input tensor
        n_hidden_layers: int
            Number of hidden layers
        n_hidden_kernels: int
            Number of kernels in each hidden layer
        n_output_channels: int
            Number of channels in output tensor
        """
        super().__init__()
        
        hidden_layers = []
        for _ in range(n_hidden_layers):
            # Add a CNN layer
            layer = nn.Conv2d(in_channels=n_input_channels, out_channels=n_hidden_kernels, kernel_size=3)
            hidden_layers.append(layer)
            # Add relu activation module to list of modules
            hidden_layers.append(nn.ReLU())
            n_input_channels = n_hidden_kernels
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Conv2d(in_channels=n_input_channels, out_channels=n_output_channels, kernel_size=3)
    
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
        # Apply hidden layers module
        hidden_features = self.hidden_layers(x)
        
        # Apply last layer (=output layer)
        output = self.output_layer(hidden_features)
        return output


# Create an instance of our CNN
cnn = CNN(n_input_channels=3, n_hidden_layers=1, n_hidden_kernels=32, n_output_channels=1)
# GPU will be much faster here
cnn.to(device=device)
# Create some input (1 sample, 3 channels, 9 height and width)
input_tensor = torch.arange(3 * 9 * 9, dtype=torch.float32, device=device).reshape((1, 3, 9, 9))
print("\nCNN")
print(f"input tensor shape: {input_tensor.shape}")
output_tensor = cnn(input_tensor)
print(f"output tensor shape: {output_tensor.shape}")

#
# Removing spatial/temporal dimensions from predictions
#

# In order to obtain a scalar output from a CNN, one can apply global pooling
# over the last CNN layer activations (e.g., maximum over all activations along
# height and width of hidden feature matrix). This approach is independent of
# the input size. Alternatively, the CNN layer output is flattened (=reshaped
# into a 1D vector) and fed into a fully-connected neural network.


################################################################################
# Multi-input CNN/NN example
################################################################################

# Let's consider a scenario with multiple input tensors per sample that we want
# to tackle with different NN types in subnetworks. To make the task more
# descriptive, assume that we have health data from patients. For each patient,
# a picture of the iris of the left eye has been taken and some additional data
# consisting of 35 features has been recorded. Furthermore, the health status of
# each patient, which will be our target to predict, is given as either 0
# (healthy) or 1 (diseased). The iris images can have different sizes but will
# always include 3 color channels. We will apply max-pooling to remove the
# spatial dimension in the image after the CNN processing, which will leave us
# with a 1D feature vector.

# To tackle this task, we can have a CNN processing the iris images and a NN
# processing the 35 features, concatenate their outputs and feed them into an
# output NN.

# Create image tensors for a minibatch of 5 samples and assume (possibly padded)
# size of 55x55:
minibatch_images = torch.arange(5 * 3 * 55 * 55, dtype=torch.float32, device=device).reshape((5, 3, 55, 55))
minibatch_features = torch.arange(5 * 35, dtype=torch.float32, device=device).reshape((5, 35))


# Create a PyTorch module "MixedNN" for our mixed NN
class MixedNN(nn.Module):
    
    def __init__(
            self,
            n_cnn_layers: int,
            n_cnn_kernels: int,
            n_subnn_layers: int,
            n_subnn_units: int,
            n_topnn_layers: int,
            n_topnn_units: int
    ):
        """Our custom NN that will process an iris image and a 35-feature vector
        
        Parameters
        ----------
        n_cnn_layers: int
            Number of layers in CNN
        n_cnn_kernels: int
            Number of kernels in each CNN layer
        n_subnn_layers: int
            Number of layers in the NN that processes the 35 features
        n_subnn_units: int
            Number of units in the NN that processes the 35 features
        n_topnn_layers: int
            Number of kernels in each hidden layer
        n_topnn_units: int
            Number of features in output tensor
        """
        super().__init__()
        n_input_channels = 3  # 3 image channels
        n_input_features = 35  # 35 input features
        n_output_units = 1  # 1 output unit
        
        # Build CNN
        cnn_layers = []
        for _ in range(n_cnn_layers):
            # Add a CNN layer
            layer = nn.Conv2d(in_channels=n_input_channels, out_channels=n_cnn_kernels, kernel_size=3)
            # Weight initialization if selu activation was used:
            # torch.nn.init.normal_(layer.weight, 0, 1 / np.sqrt(np.prod(layer.weight.shape[1:])))
            cnn_layers.append(layer)
            # Add relu activation module to list of modules
            cnn_layers.append(nn.ReLU())
            # Add max-pooling over spatial dimensions
            cnn_layers.append(nn.MaxPool2d(kernel_size=3))
            n_input_channels = n_cnn_kernels
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Build sub-NN for 35 features
        subnn_layers = []
        for _ in range(n_subnn_layers):
            # Add linear layer module to list of modules
            layer = nn.Linear(in_features=n_input_features, out_features=n_subnn_units)
            torch.nn.init.normal_(layer.weight, 0, 1 / np.sqrt(layer.in_features))
            subnn_layers.append(layer)
            # Add selu activation module to list of modules
            subnn_layers.append(nn.SELU())
            n_input_features = n_subnn_units
        self.subnn = nn.Sequential(*subnn_layers)
        
        # Build top-NN that takes concatenated CNN and sub-NN outputs
        n_input_features = n_cnn_kernels + n_subnn_units
        topnn_layers = []
        for _ in range(n_topnn_layers):
            # Add linear layer module to list of modules
            layer = nn.Linear(in_features=n_input_features, out_features=n_topnn_units)
            torch.nn.init.normal_(layer.weight, 0, 1 / np.sqrt(layer.in_features))
            topnn_layers.append(layer)
            # Add selu activation module to list of modules
            topnn_layers.append(nn.SELU())
            n_input_features = n_topnn_units
        self.topnn = nn.Sequential(*topnn_layers)
        
        self.output_layer = nn.Linear(in_features=n_topnn_units, out_features=n_output_units)
    
    def forward(self, image: torch.Tensor, features: torch.Tensor):
        """Apply CNN to ``image`` and ``features``.
        
        Parameters
        ----------
        image: torch.Tensor
            Input tensor of shape ``(n_samples, n_input_channels, height, width)``
        features: torch.Tensor
            Input tensor of shape ``(n_samples, n_input_features)``
        
        Returns
        ----------
        torch.Tensor
            Output tensor of shape ``(n_samples, n_output_features)``
        """
        # Apply CNN
        cnn_out = self.cnn(image)
        # Take the maximum over the spatial dimensions
        cnn_out, _ = cnn_out.max(dim=-1)
        cnn_out, _ = cnn_out.max(dim=-1)
        
        # Apply first NN
        subnn_out = self.subnn(features)
        
        # Concatenate outputs
        topnn_in = torch.cat([cnn_out, subnn_out], dim=-1)
        
        # Apply top NN and output layer
        topnn_out = self.topnn(topnn_in)
        output = self.output_layer(topnn_out)
        
        # Use sigmoid so that network output can be interpreted as probability
        # of target 0 or 1 (note: Due to numerical stability, this last step is
        # typically included in a corresponding loss function, see next unit)
        output = torch.sigmoid(output)
        return output


# Create an instance of our MixedNN
mnn = MixedNN(
    n_cnn_layers=3,
    n_cnn_kernels=32,
    n_subnn_layers=3,
    n_subnn_units=32,
    n_topnn_layers=3,
    n_topnn_units=16
)

# GPU will be much faster here
mnn.to(device=device)
print("\nMixedNN")
print(f"input tensor shapes: {minibatch_images.shape, minibatch_features.shape}")
output_tensor = mnn(minibatch_images, minibatch_features)
print(f"output tensor shape: {output_tensor.shape}")
