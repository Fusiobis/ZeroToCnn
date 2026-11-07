class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backprop(self, dout):
        return dout * self.mask