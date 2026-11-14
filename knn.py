import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class KNN:
    def __init__(self, k=3):
        self.k = k
    def euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_one(self, test_point):

        distances = []

        for i in range(len(self.X_train)):

            distance = self.euclidean_distance(
                test_point, self.X_train[i]
            )

            distances.append((distance, self.y_train[i]))
        distances.sort(key=lambda x: x[0])

        k_nearest = distances[:self.k]

        labels = [label for distance, label in k_nearest]

        return max(set(labels), key=labels.count)

    def predict(self, X_test):

        predictions = []

        for point in X_test:
            predictions.append(self.predict_one(point))

        return np.array(predictions)


model = KNN(k=3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("KNN Accuracy:", accuracy * 100, "%")



