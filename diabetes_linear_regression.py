# More Datasets: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

alldata = datasets.load_diabetes()
data_X = alldata.data  # Features
data_y = alldata.target  # Labels

model = LinearRegression()
model.fit(data_X, data_y)  # Train the model with the dataset

# print("Predicted Labels:")
# print(model.predict(data_X[:3, :]))
#
# print("Actual Labels:")
# print(data_y[:3])

X, y = datasets.make_regression(n_samples=150, n_features=1, n_targets=1, noise=15)
plt.scatter(X, y)
plt.show()
