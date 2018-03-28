from kNearestNeighbors import KNearestNeighbors as knn

plot = knn("../data/iris_training.csv", ["sepal_length", "sepal_width", "petal_length", "petal_width"], 5)
plot.get_category([5.0, 2.3, 3.3, 1.0])
