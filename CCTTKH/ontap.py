import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('E:/dulieutestcongcu/iris.csv')
print("Du lieu ban dau: \n", data)
data = data.values
X = data[:, :-1]
y = data[:, -1]

scaler_X = preprocessing.StandardScaler()
X_normalized = scaler_X.fit_transform(X)
scaler_y = preprocessing.LabelEncoder()
y_normalized = scaler_y.fit_transform(y)

np.set_printoptions(precision=3)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.5, random_state=24)
print("Kich thuoc du lieu train: ", X_train.shape)
print('Kich thuoc du lieu test: ', X_test.shape)
k_folds = KFold(n_splits=8, shuffle=True, random_state=24)

# Kiem tra cheo voi mo hinh naive_bayes
naive_bayes_model = GaussianNB()
f1_scores = []
precisions = []
recalls = []
accuracies = []
for train_idx, val_idx in k_folds.split(X_train):
    naive_bayes_model.fit(X_train[train_idx], y_train[train_idx])

    y_pred = naive_bayes_model.predict(X_train[val_idx])

    p, r, f, s = precision_recall_fscore_support(y_train[val_idx], y_pred, zero_division=1, average='macro')
    a = accuracy_score(y_train[val_idx], y_pred)

    accuracies.append(a)
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f)

print("Results for Naive Bayes:")
print(f"Accuracy: {accuracies}: {sum(accuracies)/len(accuracies)}")
print(f"Precision: {precisions}: {sum(precisions)/len(precisions)}")
print(f"Recall: {recalls}: {sum(recalls)/len(recalls)}")

plt.subplot(1,5,1)
plt.title("Bayes")
plt.plot(accuracies)
plt.plot(precisions)
plt.plot(recalls)
plt.legend(["Accuracies", "Precisions", "Recalls"])
plt.grid()

# Kiem tra cheo voi mo hinh SVM
svm_model = svm.SVC()
f1_scores = []
precisions = []
recalls = []
accuracies = []
for train_idx, val_idx in k_folds.split(X_train):
    svm_model.fit(X_train[train_idx], y_train[train_idx])
    y_pred = svm_model.predict(X_train[val_idx])

    p, r, f, s = precision_recall_fscore_support(y_train[val_idx], y_pred, zero_division=1, average='macro')
    a = accuracy_score(y_train[val_idx], y_pred)

    accuracies.append(a)
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f)


print("Results for SVM:")
print(f"Accuracy: {accuracies}: {sum(accuracies)/len(accuracies)}")
print(f"Precision: {precisions}: {sum(precisions)/len(precisions)}")
print(f"Recall: {recalls}: {sum(recalls)/len(recalls)}")

plt.subplot(1,5,2)
plt.title("SVM")
plt.plot(accuracies)
plt.plot(precisions)
plt.plot(recalls)
plt.legend(["Accuracies", "Precisions", "Recalls"])
plt.grid()

# Kiem tra cheo voi mo hinh KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
f1_scores = []
precisions = []
recalls = []
accuracies = []
for train_idx, val_idx in k_folds.split(X_train):
    knn_model.fit(X_train[train_idx], y_train[train_idx])

    y_pred = knn_model.predict(X_train[val_idx])

    p, r, f, s = precision_recall_fscore_support(y_train[val_idx], y_pred, zero_division=1, average='macro')
    a = accuracy_score(y_train[val_idx], y_pred)

    accuracies.append(a)
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f)

print("Results for KNN:")
print(f"Accuracy: {accuracies}: {sum(accuracies)/len(accuracies)}")
print(f"Precision: {precisions}: {sum(precisions)/len(precisions)}")
print(f"Recall: {recalls}: {sum(recalls)/len(recalls)}")

plt.subplot(1,5,3)
plt.title("KNN")
plt.plot(accuracies)
plt.plot(precisions)
plt.plot(recalls)
plt.legend(["Accuracies", "Precisions", "Recalls"])
plt.grid()

# Kiem tra cheo voi mo hinh hoi quy logistic
logistic_regression_model = LogisticRegression()
f1_scores = []
precisions = []
recalls = []
accuracies = []
for train_idx, val_idx in k_folds.split(X_train):
    logistic_regression_model.fit(X_train[train_idx], y_train[train_idx])

    y_pred = logistic_regression_model.predict(X_train[val_idx])

    p, r, f, s = precision_recall_fscore_support(y_train[val_idx], y_pred, zero_division=1, average='macro')
    a = accuracy_score(y_train[val_idx], y_pred)

    accuracies.append(a)
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f)

print("Results for logistic regression:")
print(f"Accuracy: {accuracies}: {sum(accuracies)/len(accuracies)}")
print(f"Precision: {precisions}: {sum(precisions)/len(precisions)}")
print(f"Recall: {recalls}: {sum(recalls)/len(recalls)}")

plt.subplot(1,5,4)
plt.title("logistic")
plt.plot(accuracies)
plt.plot(precisions)
plt.plot(recalls)
plt.legend(["Accuracies", "Precisions", "Recalls"])
plt.grid()

# Kiem tra cheo voi mo hinh perceptron
perceptron_model = Perceptron()
f1_scores = []
precisions = []
recalls = []
accuracies = []
for train_idx, val_idx in k_folds.split(X_train):
    perceptron_model.fit(X_train[train_idx], y_train[train_idx])

    y_pred = perceptron_model.predict(X_train[val_idx])

    p, r, f, s = precision_recall_fscore_support(y_train[val_idx], y_pred, zero_division=1, average='macro')
    a = accuracy_score(y_train[val_idx], y_pred)

    accuracies.append(a)
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f)

print("Results for Perceptron:")
print(f"Accuracy: {accuracies}: {sum(accuracies) / len(accuracies)}")
print(f"Precision: {precisions}: {sum(precisions) / len(precisions)}")
print(f"Recall: {recalls}: {sum(recalls) / len(recalls)}")

plt.subplot(1, 5, 5)
plt.title("Perceptron")
plt.plot(accuracies)
plt.plot(precisions)
plt.plot(recalls)
plt.legend(["Accuracies", "Precisions", "Recalls"])
plt.grid()
plt.show()

