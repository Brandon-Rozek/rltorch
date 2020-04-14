# rltorch
A reinforcement learning framework with the primary purpose of learning and cleaning up personal scripts.

## Installation
From GitHub
```
pip install git+https://github.com/brandon-rozek/rltorch
```

## Components
### Config
This is a dictionary that is shared around the different components. Contains hyperparameters and other configuration values.

### Environment
This component needs to support the standard openai functions reset and step.

### Network
A network takes a PyTorch nn.Module, PyTorch optimizer, and configuration.

### Target Network
Takes in a network and provides methods to sync a copy of the original network.

### Action Selector
Typtically takes in a network which it then uses to help make decisions on which actions to take.

For example, the ArgMaxSelector chooses the action that produces the highest entry in the output vector of the network.

### Memory
Stores experiences during simulations of the environment. Useful for later training.

### Agents
Takes in a network and performs some sort of training upon it.