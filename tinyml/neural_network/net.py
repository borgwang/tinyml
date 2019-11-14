class Net:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        layer_grads = []
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            layer_grads.append(layer.grads)
        return layer_grads[::-1]

    @property
    def params(self):
        return [layer.params for layer in self.layers]
