import math
import random
import time

class Matrix:
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols

        if data is not None:
            self.data = data
        else:
            self.data = [[0 for _ in range(cols)] for _ in range(rows)]

    def randomize(self, min_val=-1, max_val=1):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random.uniform(min_val, max_val)
        return self

    def add(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrix dimensions don't match for addition")

            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + other.data[i][j]
            return result
        else:
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] + other
            return result

    def subtract(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrix dimensions don't match for subtraction")

            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - other.data[i][j]
            return result
        else:
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] - other
            return result

    def multiply(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrix dimensions don't match for element-wise multiplication")

            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * other.data[i][j]
            return result
        else:
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * other
            return result

    def dot(self, other):
        if self.cols != other.rows:
            raise ValueError(f"Matrix dimensions don't match for dot product: {self.rows}x{self.cols} and {other.rows}x{other.cols}")

        result = Matrix(self.rows, other.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                sum_val = 0
                for k in range(self.cols):
                    sum_val += self.data[i][k] * other.data[k][j]
                result.data[i][j] = sum_val
        return result

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def map(self, func):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = func(self.data[i][j])
        return result

    def copy(self):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j]
        return result

    @staticmethod
    def from_array(arr):
        m = Matrix(len(arr), 1)
        for i in range(len(arr)):
            m.data[i][0] = arr[i]
        return m

    def to_array(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.data[i][j])
        return arr


class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - (x * x)

    @staticmethod
    def relu(x):
        return max(0, x)

    @staticmethod
    def relu_derivative(x):
        return 1 if x > 0 else 0

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return max(alpha * x, x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return 1 if x > 0 else alpha


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.weights_ih = Matrix(hidden_nodes, input_nodes).randomize()
        self.weights_ho = Matrix(output_nodes, hidden_nodes).randomize()

        self.bias_h = Matrix(hidden_nodes, 1).randomize()
        self.bias_o = Matrix(output_nodes, 1).randomize()

    def feedforward(self, input_array):
        inputs = Matrix.from_array(input_array)

        hidden = self.weights_ih.dot(inputs)
        hidden = hidden.add(self.bias_h)
        hidden = hidden.map(ActivationFunctions.sigmoid)

        outputs = self.weights_ho.dot(hidden)
        outputs = outputs.add(self.bias_o)
        outputs = outputs.map(ActivationFunctions.sigmoid)

        return outputs.to_array()

    def train(self, input_array, target_array):
        inputs = Matrix.from_array(input_array)

        hidden = self.weights_ih.dot(inputs)
        hidden = hidden.add(self.bias_h)
        hidden = hidden.map(ActivationFunctions.sigmoid)

        outputs = self.weights_ho.dot(hidden)
        outputs = outputs.add(self.bias_o)
        outputs = outputs.map(ActivationFunctions.sigmoid)

        targets = Matrix.from_array(target_array)

        output_errors = targets.subtract(outputs)

        output_gradients = outputs.map(ActivationFunctions.sigmoid_derivative)
        output_gradients = output_gradients.multiply(output_errors)
        output_gradients = output_gradients.multiply(self.learning_rate)

        hidden_t = hidden.transpose()
        weights_ho_deltas = output_gradients.dot(hidden_t)

        self.weights_ho = self.weights_ho.add(weights_ho_deltas)
        self.bias_o = self.bias_o.add(output_gradients)

        weights_ho_t = self.weights_ho.transpose()
        hidden_errors = weights_ho_t.dot(output_errors)

        hidden_gradients = hidden.map(ActivationFunctions.sigmoid_derivative)
        hidden_gradients = hidden_gradients.multiply(hidden_errors)
        hidden_gradients = hidden_gradients.multiply(self.learning_rate)

        inputs_t = inputs.transpose()
        weights_ih_deltas = hidden_gradients.dot(inputs_t)

        self.weights_ih = self.weights_ih.add(weights_ih_deltas)
        self.bias_h = self.bias_h.add(hidden_gradients)

    def train_batch(self, inputs_array, targets_array, epochs=1):
        for _ in range(epochs):
            for i in range(len(inputs_array)):
                self.train(inputs_array[i], targets_array[i])


class RecurrentLayer:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = Matrix(hidden_size, input_size).randomize()
        self.W_hh = Matrix(hidden_size, hidden_size).randomize()
        self.bias = Matrix(hidden_size, 1).randomize()

        self.hidden_state = Matrix(hidden_size, 1)

        self.input_states = []
        self.hidden_states = []

    def forward(self, x, reset_state=False):
        if reset_state:
            self.hidden_state = Matrix(self.hidden_size, 1)

        self.input_states.append(x.copy())
        self.hidden_states.append(self.hidden_state.copy())

        hidden = self.W_ih.dot(x)
        recurrent = self.W_hh.dot(self.hidden_state)
        self.hidden_state = hidden.add(recurrent).add(self.bias)
        self.hidden_state = self.hidden_state.map(ActivationFunctions.tanh)

        return self.hidden_state.copy()

    def reset_memory(self):
        self.input_states = []
        self.hidden_states = []


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ii = Matrix(hidden_size, input_size).randomize()
        self.W_hi = Matrix(hidden_size, hidden_size).randomize()
        self.b_i = Matrix(hidden_size, 1).randomize()

        self.W_if = Matrix(hidden_size, input_size).randomize()
        self.W_hf = Matrix(hidden_size, hidden_size).randomize()
        self.b_f = Matrix(hidden_size, 1).randomize()

        self.W_io = Matrix(hidden_size, input_size).randomize()
        self.W_ho = Matrix(hidden_size, hidden_size).randomize()
        self.b_o = Matrix(hidden_size, 1).randomize()

        self.W_ig = Matrix(hidden_size, input_size).randomize()
        self.W_hg = Matrix(hidden_size, hidden_size).randomize()
        self.b_g = Matrix(hidden_size, 1).randomize()

        self.h = Matrix(hidden_size, 1)
        self.c = Matrix(hidden_size, 1)

    def forward(self, x, reset_state=False):
        if reset_state:
            self.h = Matrix(self.hidden_size, 1)
            self.c = Matrix(self.hidden_size, 1)

        i = self.W_ii.dot(x).add(self.W_hi.dot(self.h)).add(self.b_i)
        i = i.map(ActivationFunctions.sigmoid)

        f = self.W_if.dot(x).add(self.W_hf.dot(self.h)).add(self.b_f)
        f = f.map(ActivationFunctions.sigmoid)

        o = self.W_io.dot(x).add(self.W_ho.dot(self.h)).add(self.b_o)
        o = o.map(ActivationFunctions.sigmoid)

        g = self.W_ig.dot(x).add(self.W_hg.dot(self.h)).add(self.b_g)
        g = g.map(ActivationFunctions.tanh)

        self.c = f.multiply(self.c).add(i.multiply(g))

        self.h = o.multiply(self.c.map(ActivationFunctions.tanh))

        return self.h.copy()

    def reset_state(self):
        self.h = Matrix(self.hidden_size, 1)
        self.c = Matrix(self.hidden_size, 1)


class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = Matrix(vocab_size, embedding_dim).randomize(-0.1, 0.1)

    def forward(self, word_idx):
        if word_idx >= self.vocab_size:
            raise ValueError(f"Word index {word_idx} out of range (vocab size: {self.vocab_size})")

        embedding = Matrix(self.embedding_dim, 1)
        for i in range(self.embedding_dim):
            embedding.data[i][0] = self.embeddings.data[word_idx][i]

        return embedding
