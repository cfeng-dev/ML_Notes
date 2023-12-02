import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_X = iris.data  # Features of the iris
iris_y = iris.target  # Labels of the iris

# # Example of KNN without cross validation, when k=5
# knn = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(knn, iris_X, iris_y, cv=5, scoring='accuracy')
# print(scores)
# print(scores.mean())

# Explore the effect of varying k (1-35) in KNN
k_range = range(1, 36)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)

    # Classification
    scores = cross_val_score(knn, iris_X, iris_y, cv=10, scoring='accuracy')  # Dataset is divided into 10 parts
    k_scores.append(scores.mean())
    print(f"For k={k}, the cross validated accuracy is {k_scores[k-1]}")

    # # Regression
    # loss = -cross_val_score(knn, iris_X, iris_y, cv=10, scoring='neg_mean_squared_error')
    # k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel("value of k for KNN")
plt.ylabel("cross validated accuracy")
plt.show()

# X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, random_state=4)
# knn = KNeighborsClassifier(n_neighbors=5)
#
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(knn.score(X_test,y_test))
