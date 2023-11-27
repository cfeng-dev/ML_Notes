# import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Function to count the number of misclassified predictions
def count_misclassified(actual_labels, predicted_labels):
    try:
        return sum(actual_labels != predicted_labels)
    except Exception as e:
        print(f"Unexpected error:{e}")
        return None


iris = datasets.load_iris()
iris_X = iris.data  # Features of the iris
iris_Y = iris.target  # Labels of the iris

# print(iris_X[:3, :])  # First 3 rows with all their features

# 80% of data is used for training and 20% for testing
X_training, X_testing, Y_training, Y_testing = train_test_split(iris_X, iris_Y, test_size=0.2)

knn = KNeighborsClassifier()
knn.fit(X_training, Y_training)  # Train the classifier (without model parameters)
predicted_labels = knn.predict(X_testing)
actual_labels = Y_testing

print("Predicted Labels:")
print(predicted_labels)

print("Actual Labels:")
print(actual_labels)  # For comparison

misclassified_labels = count_misclassified(actual_labels, predicted_labels)
print(f"Number of misclassified labels: {misclassified_labels}")
