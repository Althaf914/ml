import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bag.fit(X_train, y_train)
y_pred_bag = bag.predict(X_test)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

boost = AdaBoostClassifier(n_estimators=50, random_state=42)
boost.fit(X_train, y_train)
y_pred_boost = boost.predict(X_test)

acc_bag = accuracy_score(y_test, y_pred_bag)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_boost = accuracy_score(y_test, y_pred_boost)

print("Bagging Accuracy:", acc_bag)
print("Random Forest Accuracy:", acc_rf)
print("Boosting Accuracy:", acc_boost)

methods = ["Bagging", "Random Forest", "Boosting"]
accuracies = [acc_bag, acc_rf, acc_boost]

plt.bar(methods, accuracies)
plt.xlabel("Methods")
plt.ylabel("Accuracy")
plt.title("Ensemble Methods Comparison")
plt.show()