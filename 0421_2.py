from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = load_breast_cancer(as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, 
                                          test_size=0.3, random_state=1234)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scalerX = StandardScaler()
scalerX.fit(X_train)
X_train_std = scalerX.transform(X_train)
X_test_std = scalerX.transform(X_test)
clf = svm.SVC(kernel='linear')
clf.fit(X_train_std, y_train)
y_pred = clf.predict(X_test_std)
print(y_pred)
cf = confusion_matrix(y_test, y_pred)
print(cf)

print(clf.score(X_train_std, y_train))