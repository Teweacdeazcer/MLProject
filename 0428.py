from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[50,73], [65,75], [75,80], [80,82], [95,85]])
plt.scatter(X[:,0], X[:,1])
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(40, 100)
plt.ylim(60, 90)
plt.show()


# 주성분 분산
print(np.mean(X, axis=0))
print(np.var(X, axis=0))
scalerX = StandardScaler()
scalerX.fit(X)
X_std = scalerX.transform(X)
print(X_std)

pca = PCA(n_components=2)
pca.fit(X_std)

Z = pca.fit_transform(X_std)
plt.scatter(Z[:,0], Z[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()

# 주성분 로딩
loadings = pca.components_
print(loadings)

rows, columns = loadings.shape
rows_names = ['x1', 'x2']
for i in range(rows):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
               color='r', alpha=0.5)
    plt.text(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2, rows_names[i],
                color='g', ha='center', va='center')

plt.scatter(loadings[:,0], loadings[:,1], color='w')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()

