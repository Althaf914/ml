from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Correct Predictions:")
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        print("Actual:", y_test[i], "Predicted:", y_pred[i])

print("\nIncorrect Predictions:")
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print("Actual:", y_test[i], "Predicted:", y_pred[i])
