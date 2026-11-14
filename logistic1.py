import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_loss(y_true, y_prob):
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)

    loss = -np.mean(
        y_true * np.log(y_prob)
        + (1 - y_true) * np.log(1 - y_prob)
    )

    return loss


class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.1, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_prob = sigmoid(z)

            loss = compute_loss(y, y_prob)
            self.loss_history.append(loss)

            dw = np.dot(X.T, (y_prob - y)) / n_samples
            db = np.mean(y_prob - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss {loss:.6f}")

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


data = load_breast_cancer()

X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

scaler = StandardScaler()

X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

model = LogisticRegressionScratch(learning_rate=0.1, epochs=10000)

model.fit(X_train_standardized, Y_train)

Y_pred = model.predict(X_test_standardized)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

precision = precision_score(Y_test, Y_pred)
print("Precision:", precision)

recall = recall_score(Y_test, Y_pred)
print("Recall:", recall)

f1score = f1_score(Y_test, Y_pred)
print("F1 Score:", f1score)