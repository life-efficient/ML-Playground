from linear_models import LinearRegression
from data import get_regression_data
from vis import visualise_regression_data

X, Y = get_regression_data()
print(X.shape)
print(Y.shape)
visualise_regression_data(X, Y)
# linearModel = LinearRegression()