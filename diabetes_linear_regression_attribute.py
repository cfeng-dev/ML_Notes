# More Datasets: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets

from sklearn import datasets
from sklearn.linear_model import LinearRegression

alldata = datasets.load_diabetes()
data_X = alldata.data  # Features
data_y = alldata.target  # Labels

model = LinearRegression()
model.fit(data_X, data_y)  # Train the model with the dataset

print("Model coefficients:")
print(model.coef_)

print("\nModel intercept:")
print(model.intercept_)  # Expected mean value of Y when all X=0

print("\nModel parameters:")
print(model.get_params())  # Parameters of the model

print("\nModel performance score (R^2):")
print(model.score(data_X, data_y))  # Coefficient of determination
