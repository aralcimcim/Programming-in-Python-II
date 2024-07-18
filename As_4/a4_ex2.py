import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: list,
                 use_batchnormalization: bool,
                 num_classes: int,
                 kernel_size: list,
                 activation_function: nn.Module = nn.ReLU()
                 ):
        
        super(SimpleCNN, self).__init__()

        #input dims: (1, 3, 70, 100) -> (32, 3, 70, 100)
        #                               (64, 3, 70, 100)
        #                               (128, 3, 70, 100)
        #                               (10, 3, 70, 100)

        # for kernel_size = 3, padding = 1

        self.convo1 = nn.Conv2d(input_channels, hidden_channels[0], kernel_size[0], 
                               padding=(kernel_size[0] // 2, kernel_size[0] // 2))
        
        self.convo2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size[1],
                               padding=(kernel_size[1] // 2, kernel_size[1] // 2))
        
        self.convo3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size[2],
                               padding=(kernel_size[2] // 2, kernel_size[2] // 2))
        
        # fully connected 
        self.fc = nn.Linear(hidden_channels[2] * 70 * 100, num_classes)

        self.activation_function = activation_function

        self.use_batchnormalization = use_batchnormalization

        if use_batchnormalization:
            self.batch_n1 = nn.BatchNorm2d(hidden_channels[0])
            self.batch_n2 = nn.BatchNorm2d(hidden_channels[1])
            self.batch_n3 = nn.BatchNorm2d(hidden_channels[2])

    def forward(self, input_images: torch.Tensor):
        x = self.convo1(input_images)
        if self.use_batchnormalization:
            x = self.batch_n1(x)
        x = self.activation_function(x)

        x = self.convo2(x)
        if self.use_batchnormalization:
            x = self.batch_n2(x)
        x = self.activation_function(x)

        x = self.convo3(x)
        if self.use_batchnormalization:
            x = self.batch_n3(x)
        x = self.activation_function(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# if __name__ == "__main__":
#     torch.random.manual_seed(1234)
#     network = SimpleCNN(3, [32, 64, 128], True, 10, [3, 5, 7], activation_function=nn.ELU())
#     input = torch.randn(1, 3, 70, 100, requires_grad = False)
#     output = network(input)
#     print(output)