from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[2, 3], [1, 2]])
y_train = np.array([1, -1])

X_test = np.array([[3, 3]])

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', marker='D', s=100)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)