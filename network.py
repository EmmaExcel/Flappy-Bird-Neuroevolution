import numpy as np


class Network:
    def __init__(self, topology):
        self.topology = list(topology)
        self.weights = []
        self.biases = []
        for i in range(len(self.topology) - 1):
            fan_in = self.topology[i]
            fan_out = self.topology[i + 1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, inputs):
        x = np.array(inputs, dtype=np.float64)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)
            else:
                x = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
        return x

    def get_params(self):
        parts = [w.flatten() for w in self.weights] + [b.flatten() for b in self.biases]
        return np.concatenate(parts)

    def set_params(self, params):
        params = np.array(params, dtype=np.float64)
        idx = 0
        for i in range(len(self.topology) - 1):
            size = self.topology[i] * self.topology[i + 1]
            self.weights[i] = params[idx:idx + size].reshape(
                self.topology[i], self.topology[i + 1]
            )
            idx += size
        for i in range(len(self.topology) - 1):
            size = self.topology[i + 1]
            self.biases[i] = params[idx:idx + size].copy()
            idx += size

    def param_count(self):
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)

    def clone(self):
        net = Network(self.topology)
        net.set_params(self.get_params().copy())
        return net
