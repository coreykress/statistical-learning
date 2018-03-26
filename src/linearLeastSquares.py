import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearLeastSquares():

    def __init__(self, file_path, feature_columns, iterations, learning_rate):
        self.file_path = file_path
        self.data_frame = pd.read_csv(file_path, names=feature_columns)
        self.X = self.data_frame.loc[:, feature_columns]
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.ones = np.ones([self.X.shape[0], 1])

    def display_one_dimensional(self, x_col, y_col):
        x_data = self.data_frame.loc[:, x_col].values.reshape(-1, 1)
        y_data = self.data_frame.loc[:, y_col].values.reshape(-1, 1)
        x_data = np.concatenate([self.ones, x_data], 1)
        print(x_data)
        theta = np.array([[0.0, 0.0]])
        g, cost = self.gradient_descent(x_data, y_data, theta, self.learning_rate, self.iterations)

        plt.scatter(x_data[:, 1], y_data)
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = g[0][0] + g[0][1] * x_vals
        plt.plot(x_vals, y_vals, '--')
        plt.show()

    def display_multi_variable_regression(self):
        norm_data = self.normalize_data(self.X)
        norm_data = np.concatenate([self.ones, norm_data], 1)

        x_data = norm_data[:, :-1]
        y_data = norm_data[:, 2:3]
        theta = np.zeros([1, 3])

        g, cost = self.gradient_descent(x_data, y_data, theta, self.learning_rate, self.iterations)

    def compute_cost(self, x_data, y, theta):
        inner = np.power(((x_data.dot(theta.transpose())) - y), 2)
        return np.sum(inner) / (2 * len(x_data))

    def normalize_data(self, data):
        data = (data - data.mean()) / data.std()
        return data;

    def gradient_descent(self, x_data, y_data, theta, alpha, iterations):
        cost = np.zeros(iterations)
        for i in range(iterations):
            alpha_len = alpha / len(x_data)
            theta = theta - alpha_len * (np.sum(x_data * (x_data.dot(theta.transpose()) - y_data)))
            cost[i] = self.compute_cost(x_data, y_data, theta)
            if i % 10 == 0:
                print(cost[i])

        return theta, cost
