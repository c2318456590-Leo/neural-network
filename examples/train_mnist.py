import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src import Sequential, Dense, ReLU, Softmax
from src import CrossEntropy, Adam

print("MNIST")
print("-" * 40)

print("loading data...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"train: {X_train.shape[0]}, test: {X_test.shape[0]}")

model = Sequential()
model.add(Dense(784, 256))
model.add(ReLU())
model.add(Dense(256, 128))
model.add(ReLU())
model.add(Dense(128, 10))
model.add(Softmax())

print("training...")
losses, accuracies = model.train(
    X_train, y_train,
    epochs=20, batch_size=64,
    optimizer=Adam(learning_rate=0.001),
    loss_fn=CrossEntropy(),
    verbose=True
)

print("-" * 40)
train_acc = model.evaluate(X_train, y_train)
test_acc = model.evaluate(X_test, y_test)
print(f"train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")
