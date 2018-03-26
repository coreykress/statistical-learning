import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearLeastSquares():

    def __init__(self, file_path, iterations, learning_rate):
        self.file_path = file_path
        self.data_frame = pd.read_csv(file_path)
        self.X = self.data_frame.loc[:, ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.ones = np.ones([self.X.shape[0], 1])

    def display_one_dimensional(self, x_col, y_col):
        x_data = self.data_frame.loc[:, x_col].values.reshape(-1, 1)
        y_data = self.data_frame.loc[:, y_col].values.reshape(-1, 1)
        x_data = np.concatenate([self.ones, x_data], 1)
        theta = np.array([[0.0, 0.0]])
        g, cost = self.gradient_descent(x_data, y_data, theta, self.learning_rate, self.iterations)

        plt.scatter(x_data[:, 1], y_data)
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = g[0][0] + g[0][1] * x_vals
        plt.plot(x_vals, y_vals, '--')
        plt.show()

    def compute_cost(self, x_data, y, theta):
        inner = np.power(((x_data.dot(theta.transpose())) - y), 2)
        return np.sum(inner) / (2 * len(x_data))

    def gradient_descent(self, x_data, y_data, theta, alpha, iterations):
        cost = 0
        for i in range(iterations):
            alpha_len = alpha / len(x_data)
            theta -= alpha_len * (np.sum(x_data.dot(theta.transpose()) - y_data * x_data, axis=0))
            cost = self.compute_cost(x_data, y_data, theta)
            if i % 10 == 0:
                print(cost)

        return theta, cost
