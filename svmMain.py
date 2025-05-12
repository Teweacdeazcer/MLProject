from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = load_breast_cancer(as_frame=True)

# 데이터 분할(훈련 70%, 테스트 30%)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, 
                                          test_size=0.3, random_state=1234)

scalerX = StandardScaler()
scalerX.fit(X_train)
X_train_std = scalerX.transform(X_train)
X_test_std = scalerX.transform(X_test)


clf = svm.SVC(kernel='linear')
clf.fit(X_train_std, y_train)
y_pred = clf.predict(X_test_std)

best_c = None
best_acc = 0
best_y_pred = None
c_list = [0.001, 0.01, 0.1, 1, 10, 100]

for c in c_list:
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    acc = accuracy_score(y_test, y_pred)

    print(f"C = {c:<6} → 정확도: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_c = c
        best_y_pred = y_pred

print(f"\n최적의 C 값은 {best_c} (정확도: {best_acc:.4f})")

# 최적의 c일 때의 confusion matrix
cf = confusion_matrix(y_test, best_y_pred)
print("\nConfusion Matrix (Best C):")
print(cf)

# classification report
print("\nClassification Report:")
print(classification_report(y_test, best_y_pred, target_names=data.target_names))

# confusion matrix 시각화
plt.figure(figsize=(6, 5))
sns.heatmap(cf, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(f"Confusion Matrix (C={best_c})")
plt.xticks(ticks=[0.5, 1.5], labels=["Malignant", "Benign"])
plt.yticks(ticks=[0.5, 1.5], labels=["Malignant", "Benign"])
plt.tight_layout()
plt.show()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 대표 feature 4개 선택
selected_features = ['mean radius', 'mean texture', 'mean smoothness', 'mean concavity', 'target']
df_pair = df[selected_features]

# pairplot
sns.pairplot(df_pair, hue='target', palette='Set1')
plt.suptitle("Pairplot of Original Features (Target = Malignant[0], Benign[1])", y=1.02)
plt.show()

'''# PCA'''
print("\n# PCA")
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# 누적 설명 분산 비율 시각화
pca = PCA()
pca.fit(X)
explained = np.cumsum(pca.explained_variance_ratio_)
plt.plot(np.arange(1, len(explained)+1), explained, marker='o')
plt.axhline(0.95, color='r', linestyle='--')
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.show() # 주성분 2개로 98% 설명 가능

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
# pca = PCA(n_components=2) # 현재 주성분 2개
pca = PCA(n_components=3) # 현재 주성분 3개
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# SVM 훈련 (PCA 후)
clf = svm.SVC(kernel='linear')
clf.fit(X_train_pca, y_train)

# 예측 및 정확도
y_pred = clf.predict(X_test_pca)
c_list_pca = [0.001, 0.01, 0.1, 1, 10, 100]
best_acc_pca = 0
best_c_pca = None
best_y_pred_pca = None

for c in c_list_pca:
    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"C = {c:<6} → PCA 후 정확도: {acc:.4f}")

    if acc > best_acc_pca:
        best_acc_pca = acc
        best_c_pca = c
        best_y_pred_pca = y_pred

print(f"\nPCA 후 최적의 C 값은 {best_c_pca} (정확도: {best_acc_pca:.4f})")

# confusion matrix 출력
cm = confusion_matrix(y_test, best_y_pred_pca)
print("\nConfusion Matrix (Best C):")
print(cm)

# confusion matrix 시각화
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title(f"Confusion Matrix (C={best_c_pca})")
plt.xticks(ticks=[0.5, 1.5], labels=["Malignant", "Benign"])
plt.yticks(ticks=[0.5, 1.5], labels=["Malignant", "Benign"])
plt.show()

# classification report 출력
print("\nClassification Report:")
print(classification_report(y_test, best_y_pred_pca, target_names=data.target_names))

# PCA pairplot
X_total_pca = pca.transform(scaler.transform(X))  
# df_pca = pd.DataFrame(X_total_pca, columns=['PCA1', 'PCA2']) # 주성분 2개일 때
df_pca = pd.DataFrame(X_total_pca, columns=['PCA1', 'PCA2', 'PCA3']) # 주성분 3개일 때
df_pca['target'] = y

# PCA pairplot 시각화
sns.pairplot(df_pca, hue='target', palette='Set1')
plt.show()