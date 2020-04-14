import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# This class utilizes this property of the normal distribution
# N(mu, sigma) = mu + sigma * N(0, 1)
class NoisyLinear(nn.Linear):
    """
    Draws the parameters of nn.Linear from a normal distribution.
    The parameters of the normal distribution are registered as
    learnable parameters in the neural network.
    Parameters
    ----------
    in_features
      Size of each input sample.
    out_features
      Size of each output sample.
    sigma_init
      The starting standard deviation of guassian noise.
    bias
       If set to False, the layer will not
       learn an additive bias.
       Default: True
    """
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # One of the parameters the network is going to tune is the 
        # standard deviation of the gaussian noise on the weights
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        # Reserve space for N(0, 1) of weights in the forward() call
        self.register_buffer("s_normal_weight", torch.zeros(out_features, in_features))
        if bias:
            # If a bias exists, then we manipulate the standard deviation of the
            # gaussion noise on them as well
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
            # Reserve space for N(0, 1) of bias in the foward() call
            self.register_buffer("s_normal_bias", torch.zeros(out_features))
        self.reset_parameters()
  
    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        nn.init.uniform_(self.bias, -std, std)
  
    def forward(self, x):
        r"""
        Calculates the output :math:`y` through the following:

        :math:`sigma \sim N(mu_1, std_1)`

        :math:`bias \sim N(mu_2, std_2)`

        :math:`y = sigma \cdot x + bias`
        """
    # Fill s_normal_weight with values from the standard normal distribution
        self.s_normal_weight.normal_()
        weight_noise = self.sigma_weight * self.s_normal_weight.clone().requires_grad_()

        bias = None
        if self.bias is not None:
            # Fill s_normal_bias with values from standard normal
            self.s_normal_bias.normal_()
            bias = self.bias + self.sigma_bias * self.s_normal_bias.clone().requires_grad_()

        return F.linear(x, self.weight + weight_noise, bias)
