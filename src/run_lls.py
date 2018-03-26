from linearLeastSquares import LinearLeastSquares as lls

"""
a fairly decent linear regression

iterations = 1000
learning rate = .00003
thetas = [0,0]
"""
plot = lls("../data/iris_training.csv", 1000, .00003)
plot.display_one_dimensional("sepal_length", "sepal_width")
