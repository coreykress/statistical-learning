from linearLeastSquares import LinearLeastSquares as lls

"""
a fairly decent linear regression

iterations = 1000
learning rate = .00003
thetas = [0,0]
["sepal_length", "sepal_width", "petal_length", "petal_width"]
"""
plot = lls("../data/iris_training.csv", ["sepal_length", "sepal_width", "petal_length", "petal_width"], 1000, .00003)
plot.display_one_dimensional("sepal_length", "sepal_width")


"""
multivariable linear regression

iterations
learning_rate
thetas
[]
"""
# plot = lls("../data/homes.txt", ["size","bedroom","price"], 1000, .01)
# plot.display_multi_variable_regression()
