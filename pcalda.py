import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA Transformed Train Data:")
print(X_train_pca[:5])

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

print("\nLDA Transformed Train Data:")
print(X_train_lda[:5])

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)
plt.title("PCA Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train)
plt.title("LDA Visualization")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.show()
