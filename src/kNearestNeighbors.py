import numpy as np
import pandas as pd


class KNearestNeighbors:
    def __init__(self, file_path, feature_columns, k):
        self.k = k
        self.dataFrame = pd.read_csv(
            file_path,
            header=0,
            names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
            usecols=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        ).convert_objects(convert_numeric=True)
        self.trainingData = []

    def get_category(self, test_data):
        k_neighbors = []
        for index, row in self.dataFrame.iterrows():
            distance = self.euclidean_distance(test_data, row[:-1])
            k_neighbors.append((distance, row[-1]))
        k_neighbors = sorted(k_neighbors)[:self.k]
        count_types = {}
        for group in k_neighbors:
            if group[1] not in count_types:
                count_types[group[1]] = 1
                continue
            count_types[group[1]] += 1
        v = list(count_types.values())
        k = list(count_types.keys())
        output = k[v.index(max(v))]
        print(output)
        return output

    @staticmethod
    def euclidean_distance(x, y):
        """
        square root of the sums of the differences squared
        sqrt(sum(((x-y)^2))
        """
        distance_sum = 0
        for i in range(1, len(x)):
            distance_sum += pow((x[i] - y[i]), 2)
        return np.sqrt(distance_sum)
