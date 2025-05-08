from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 학습
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 예측
y_pred = knn.predict(X_test)

# 정확도
acc_original = accuracy_score(y_test, y_pred) # 100%

# 시각화
plt.figure(figsize=(12,8))
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("Original Data (Iris)")
plt.colorbar()
plt.show()

# histogram
feature_names = iris.feature_names

plt.figure(figsize=(12,8))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(X[:,i], bins=20, color='skyblue', edgecolor='black')
    plt.title(feature_names[i])

plt.tight_layout()
plt.show()


# PCA 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 학습
knn_pca = KNeighborsClassifier(n_neighbors=3)
knn_pca.fit(X_train_pca, y_train_pca)

# 예측
y_pred_pca = knn_pca.predict(X_test_pca)

# 정확도
acc_pca = accuracy_score(y_test_pca, y_pred_pca) # 100%

# 시각화
plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA result (Iris)")
plt.colorbar()
plt.show()

# histogram
plt.figure(figsize=(8,4))

for i in range(2):
    plt.subplot(1,2,i+1)
    plt.hist(X_pca[:,i], bins=20, color='lightcoral', edgecolor='black')
    plt.title(f"PCA Component {i+1}")

plt.tight_layout()
plt.show()

# 결과 비교
print(f"Original Data Accuracy: {acc_original:.2f}")
print(f"PCA Data Accuracy: {acc_pca:.2f}")