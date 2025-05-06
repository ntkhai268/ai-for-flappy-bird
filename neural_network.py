import numpy as np

class NeuralNetwork:
    def __init__(self, weights: list[float], hidden_size: int = 4):
        self.hidden_size = hidden_size
        self.input_size = 6  # 8 đặc trưng + 1 bias
        self.output_size = 1

        # Tách trọng số thành 2 phần:
        # 1. input → hidden (6 × hidden_size)
        # 2. hidden → output (hidden_size)
        split = self.input_size * self.hidden_size
        self.w1 = np.array(weights[:split]).reshape((self.input_size, self.hidden_size))
        self.w2 = np.array(weights[split:split + self.hidden_size]).reshape((self.hidden_size, self.output_size))

    def activate(self, x):
        return 1 / (1 + np.exp(-x))  # sigmoid

    def predict(self, inputs):
        # Thêm bias vào đầu vào
        inputs = np.array(inputs + [1.0])  # bias
        h = self.activate(np.dot(inputs, self.w1))       # lớp ẩn
        output = self.activate(np.dot(h, self.w2))       # lớp output
        return output[0] > 0.5  # chim nhảy nếu xác suất cao
