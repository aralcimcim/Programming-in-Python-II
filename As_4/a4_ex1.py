import torch.nn as nn
import torch

class SimpleNetwork(nn.Module):
    def __init__(self,
                 input_neurons: int,
                 hidden_neurons: int,
                 output_neurons: int,
                 use_bias: bool,
                 activation_function: nn.Module = nn.ReLU()
                 ):
        
        super(SimpleNetwork, self).__init__()
        # dims: (40, 10) -> (10, 20) -> (20, 30) -> (30, 5)
        self.input_layer = nn.Linear(input_neurons, hidden_neurons[0], bias=use_bias)
        self.hidden_1 = nn.Linear(hidden_neurons[0], hidden_neurons[1], bias=use_bias)
        self.hidden_2 = nn.Linear(hidden_neurons[1], hidden_neurons[2], bias=use_bias)
        self.output_layer = nn.Linear(hidden_neurons[2], output_neurons, bias=use_bias)
        self.activation_function = activation_function
    # (1, 40) -> (1, 5) end
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_function(self.input_layer(x))
        x = self.activation_function(self.hidden_1(x))
        x = self.activation_function(self.hidden_2(x))
        # no activation
        x = self.output_layer(x)
        return x
    
# if __name__ == "__main__":
#     torch.random.manual_seed(1234)
#     simple_network = SimpleNetwork(40, [10, 20, 30], 5, True)
#     input = torch.randn(1, 40, requires_grad = False)
#     output = simple_network(input)
#     print(output)