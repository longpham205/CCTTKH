import os
root_dir = os.path.abspath(os.path.dirname(__file__))
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay,  confusion_matrix
from scipy.fftpack import dct

RANDOM_STATE = 42      
TEST_SIZE    = 0.25    
K_FOLD       = 3      
CV           = 5      


df = pd.read_csv(os.path.join(root_dir,'Wine.csv'))
    
X = df.drop('class', axis = 1).values
y_raw = df['class']
print(y_raw[5])

Y = LabelEncoder().fit_transform(y_raw) 
x_scale = MinMaxScaler().fit_transform(X)
x_dct = dct(x_scale, norm = 'ortho')


X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size    = TEST_SIZE,
                                                    random_state = RANDOM_STATE)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("Y_train: ", Y_train.shape)
print("Y_test: ", Y_test.shape)

kf = KFold(n_splits=K_FOLD, shuffle=False)
df_y = np.array(Y)
i = 1
X_train_np = []
X_test_np = []
Y_train_np = []
Y_test_np = []
for train_index, test_index in kf.split(X):
      X_train, X_test = X[train_index], X[test_index]
      Y_train, Y_test = Y[train_index], Y[test_index]
      X_train_np.append(X_train)
      X_test_np.append(X_test)
      Y_train_np.append(Y_train)
      Y_test_np.append(Y_test)
      
#Gaus
from sklearn.naive_bayes import GaussianNB

accuracies_gauss = []
precisions_gauss = []
recalls_gauss    = []
f1_scores_gauss  = []
train_times_gauss = []


for i in range(K_FOLD):
    X_train, X_test = X_train_np[i], X_test_np[i]
    Y_train, Y_test = Y_train_np[i], Y_test_np[i]

    model_gauss = GaussianNB()

    start = time.time()
    
    model_gauss.fit(X_train, Y_train)
    finish = time.time() - start
    print('Thời gian huấn luyện :', finish)

    y_pred_gauss = model_gauss.predict(X_test)

    accuracy_gauss  = accuracy_score(Y_test, y_pred_gauss)
    precision_gauss = precision_score(Y_test, y_pred_gauss, average='macro')
    recall_gauss    = recall_score(Y_test, y_pred_gauss, average='macro')
    f1score_gauss   = f1_score(Y_test, y_pred_gauss, average='macro')

    print("FOLD:", i)
    print("{:15}: {:>5.4}".format('accuracy', accuracy_gauss))
    print("{:15}: {:>5.4}".format('precision', precision_gauss))
    print("{:15}: {:>5.4}".format('recall', recall_gauss))
    print("{:15}: {:>5.4}".format('f1score', f1score_gauss))
    print("{:15}: {:>5.4}".format('time', finish))

    print("######################################")

    accuracies_gauss.append(accuracy_gauss)
    precisions_gauss.append(precision_gauss)
    recalls_gauss.append(recall_gauss)
    f1_scores_gauss.append(f1score_gauss)
    train_times_gauss.append(finish)
    i+=1

mean_accuracy_gauss = np.mean(accuracies_gauss)
std_acc_gauss      = np.std(accuracies_gauss)

mean_precision_gauss = np.mean(precisions_gauss)
std_pre_gauss       = np.std(precisions_gauss)

mean_recall_gauss = np.mean(recalls_gauss)
std_rec_gauss       = np.std(recalls_gauss)

mean_f1score_gauss = np.mean(f1_scores_gauss)
std_f1_gauss       = np.std(f1_scores_gauss)

mean_time_gauss = np.mean(train_times_gauss)
std_time_gauss  = np.std(train_times_gauss)

print("{:15}: {:>5.4} ".format('mean_accuracy', mean_accuracy_gauss, std_acc_gauss))
print("{:15}: {:>5.4} ".format('mean_precision', mean_precision_gauss, std_pre_gauss))
print("{:15}: {:>5.4} ".format('mean_recall', mean_recall_gauss, std_rec_gauss))
print("{:15}: {:>5.4} ".format('mean_f1_score', mean_f1score_gauss, std_f1_gauss))
print("{:15}: {:>5.4} ".format('mean_time', mean_time_gauss, std_time_gauss))

Label = ['accuracy', 'precision', 'recall', 'f1score', 'time']
Value = [mean_accuracy_gauss, mean_precision_gauss, mean_recall_gauss, mean_f1score_gauss, mean_time_gauss]
color = ['blue', 'red', 'green', 'yellow', 'orange']
plt.bar(Label, Value, color=color)
plt.title('Biểu đồ đánh giá mô hình Bayes')
plt.show()

#Svm
accuracies_svm = []
precisions_svm = []
recalls_svm    = []
f1_scores_svm  = []
train_times_svm = []


for i in range(K_FOLD):
    X_train, X_test = X_train_np[i], X_test_np[i]
    Y_train, Y_test = Y_train_np[i], Y_test_np[i]

    model_svm = SVC()

    start = time.time()

    model_svm.fit(X_train, Y_train)
    finish = time.time() - start
    print('Thời gian huấn luyện :', finish)

    y_pred_svm = model_svm.predict(X_test)

    accuracy_svm  = accuracy_score(Y_test, y_pred_svm)
    precision_svm = precision_score(Y_test, y_pred_svm, average='macro')
    recall_svm    = recall_score(Y_test, y_pred_svm, average='macro')
    f1score_svm   = f1_score(Y_test, y_pred_svm, average='macro')

    print("FOLD:", i)
    print("{:15}: {:>5.4}".format('accuracy', accuracy_svm))
    print("{:15}: {:>5.4}".format('precision', precision_svm))
    print("{:15}: {:>5.4}".format('recall', recall_svm))
    print("{:15}: {:>5.4}".format('f1score', f1score_svm))
    print("{:15}: {:>5.4}".format('time', finish))

    print("######################################")

    accuracies_svm.append(accuracy_svm)
    precisions_svm.append(precision_svm)
    recalls_svm.append(recall_svm)
    f1_scores_svm.append(f1score_svm)
    train_times_svm.append(finish)
    i+=1

mean_accuracy_svm = np.mean(accuracies_svm)
std_acc_svm      = np.std(accuracies_svm)

mean_precision_svm = np.mean(precisions_svm)
std_pre_svm      = np.std(precisions_svm)

mean_recall_svm = np.mean(recalls_svm)
std_rec_svm       = np.std(recalls_svm)

mean_f1score_svm = np.mean(f1_scores_svm)
std_f1_svm       = np.std(f1_scores_svm)

mean_time_svm = np.mean(train_times_svm)
std_time_svm  = np.std(train_times_svm)

print("{:15}: {:>5.4} ".format('mean_accuracy', mean_accuracy_svm, std_acc_svm))
print("{:15}: {:>5.4} ".format('mean_precision', mean_precision_svm, std_pre_svm))
print("{:15}: {:>5.4} ".format('mean_recall', mean_recall_svm, std_rec_svm))
print("{:15}: {:>5.4} ".format('mean_f1_score', mean_f1score_svm, std_f1_svm))
print("{:15}: {:>5.4} ".format('mean_time', mean_time_svm, std_time_svm))

Label = ['accuracy', 'precision', 'recall', 'f1score', 'time']
Value = [mean_accuracy_svm, mean_precision_svm, mean_recall_svm, mean_f1score_svm, mean_time_svm]
color = ['blue', 'red', 'green', 'yellow', 'orange']
plt.bar(Label, Value, color=color)
plt.title('Biểu đồ đánh giá mô hình SVM')
plt.show() 
  
#Logistic
from sklearn.linear_model import LogisticRegression

accuracies_log = []
precisions_log = []
recalls_log    = []
f1_scores_log  = []
train_times_log = []

for i in range(K_FOLD):

    X_train = X_train_np[i]
    X_test  = X_test_np[i]
    Y_train = Y_train_np[i]
    Y_test  = Y_test_np[i]

    model_log = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')

    start = time.time()
    model_log.fit(X_train, Y_train)
    finish = time.time() - start
    print('Thời gian huấn luyện :', finish)

    y_pred_log = model_log.predict(X_test)

    accuracy_log  = accuracy_score(Y_test, y_pred_log)
    precision_log = precision_score(Y_test, y_pred_log, average='macro', zero_division=0)
    recall_log    = recall_score(Y_test, y_pred_log, average='macro', zero_division=0)
    f1score_log   = f1_score(Y_test, y_pred_log, average='macro', zero_division=0)

    print("FOLD:", i)
    print("{:15}: {:>5.4f}".format('accuracy', accuracy_log))
    print("{:15}: {:>5.4f}".format('precision', precision_log))
    print("{:15}: {:>5.4f}".format('recall', recall_log))
    print("{:15}: {:>5.4f}".format('f1score', f1score_log))
    print("{:15}: {:>5.4f}".format('time', finish))
    print("######################################")

    accuracies_log.append(accuracy_log)
    precisions_log.append(precision_log)
    recalls_log.append(recall_log)
    f1_scores_log.append(f1score_log)
    train_times_log.append(finish)

mean_accuracy_log = np.mean(accuracies_log)
std_acc_log       = np.std(accuracies_log)

mean_precision_log = np.mean(precisions_log)
std_pre_log        = np.std(precisions_log)

mean_recall_log = np.mean(recalls_log)
std_rec_log     = np.std(recalls_log)

mean_f1score_log = np.mean(f1_scores_log)
std_f1_log       = np.std(f1_scores_log)

mean_time_log = np.mean(train_times_log)
std_time_log  = np.std(train_times_log)

print("{:15}: {:>7.4f} (std {:>7.4f})".format('mean_accuracy', mean_accuracy_log, std_acc_log))
print("{:15}: {:>7.4f} (std {:>7.4f})".format('mean_precision', mean_precision_log, std_pre_log))
print("{:15}: {:>7.4f} (std {:>7.4f})".format('mean_recall', mean_recall_log, std_rec_log))
print("{:15}: {:>7.4f} (std {:>7.4f})".format('mean_f1_score', mean_f1score_log, std_f1_log))
print("{:15}: {:>7.4f} (std {:>7.4f})".format('mean_time', mean_time_log, std_time_log))


