import numpy as np
from keras.datasets import cifar10 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import KFold #

num_folds = 10
no_epochs = 10 
batch_size = 32 
no_classes = 10 

def def_load_data() :
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_test = X_test / 255
    X_train = X_train / 255

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    return X, y

def def_get_model():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                     input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="Adam",
                  metrics=['accuracy'])

    return model

accuracy_list = []
loss_list = []
fold_idx = 1

X, y = def_load_data()

kfold = KFold(n_splits=num_folds, shuffle=True)

for train_ids, val_ids in kfold.split(X, y):
    model = def_get_model()

    print(f"Bắt đầu train Fold {fold_idx}")

    model.fit(X[train_ids], y[train_ids],
              batch_size=batch_size,
              epochs=no_epochs, 
              verbose=1)

    scores = model.evaluate(X[val_ids], y[val_ids], verbose=0)

    print(f"Đã train xong Fold {fold_idx}")

    accuracy_list.append(scores[1] * 100)
    loss_list.append(scores[0])

    fold_idx = fold_idx + 1

print('** Chi tiết các fold')
for i in range(0, len(accuracy_list)):
    print(f'-> Fold {i+1} - Loss: {loss_list[i]:.4f} - Accuracy: {accuracy_list[i]:.2f}%') # In chi tiết từng fold

print('* Đánh giá tổng thể các folds:')
print(f'> Accuracy: {np.mean(accuracy_list):.4f} (Độ lệch +- {np.std(accuracy_list):.4f})')
print(f'> Loss: {np.mean(loss_list):.4f}')