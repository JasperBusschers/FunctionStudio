
import torch.nn as nn
class NeuralNetwork:
    """ A modular neural network class for easy customization and extension. """

    def __init__(self, input_size, output_size, hidden_size=4, depth=1):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.model = self._build_model()

    def _build_model(self):
        layers = [nn.Linear(self.input_size, self.hidden_size), nn.ReLU()]
        for _ in range(self.depth - 1):
            layers.extend([nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()])
        if self.output_size==1:
            layers.extend([nn.Linear(self.hidden_size, self.output_size), nn.Sigmoid()])
        else:
            layers.extend([nn.Linear(self.hidden_size ,self.output_size) ,nn.Softmax()])
        return nn.Sequential(*layers)

    def set_weights(self, weights):
        start = 0
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                for param in [layer.weight, layer.bias]:
                    end = start + param.numel()
                    param.data = weights[start:end].view(param.size())
                    start = end

    def evaluate(self, x, y):
        pred = self.model(x)
        loss = nn.BCELoss()
        return loss(pred, y).item()

    def get_solution_length(self):
        return sum(param.numel() for param in self.model.parameters())
