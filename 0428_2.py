from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = datasets.load_iris()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
print(df.head())

plt.figure(figsize=(8, 8))
ax = sns.pairplot(df, hue='target')
plt.show()

X_df = df.iloc[:, [0, 1, 2, 3]]
print(X_df.mean(axis=0))
print(X_df.var(axis=0))

scalerX = StandardScaler()
scalerX.fit(data.data)
X_std = scalerX.transform(data.data)
print(X_std)

pca=PCA()
pca.fit(X_std)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

plt.plot(pca.explained_variance_, 'o-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.show()

Z = pca.fit_transform(X_std)
Z_df = pd.DataFrame(Z, columns=['PC1', 'PC2', 'PC3', 'PC4'])
print(Z_df.head())

Z_df['target'] = df['target']
plt.figure(figsize=(8, 8))
ax = sns.pairplot(df, hue='target')
plt.show()

loadings = pca.components_
print(loadings)

rows, columns = loadings.shape
rows_names = ['x1', 'x2', 'x3', 'x4']
for i in range(rows):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
             color='r', alpha=0.5)
    plt.text(loadings[i, 0] * 1.2, loadings[i, 1] * 1.2, rows_names[i],
             color='g', ha='center', va='center')
plt.scatter(loadings[:, 0], loadings[:, 1], color='w')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()