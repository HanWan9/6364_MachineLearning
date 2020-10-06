import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import csv
import os
from sklearn.model_selection import train_test_split
from random import randrange
from collections import defaultdict
from multicpu import multi_cpu



# load the data using numpy
def load_mnist_data(train_name,test_name):
    train = pd.read_csv(train_name)
    print("training data:", train.shape)
    x_train = train.values[:,1:]
    y_train = train.values[:,0]

    test = pd.read_csv(test_name)
    print("test data:", test.shape)
    x_test = test.values[:, 1:]
    y_test = test.values[:, 0]
    return x_train, y_train, x_test, y_test


def knn(inx,dataset,labels,k):
    dist = (((dataset-inx)**2).sum(1))**0.5
    sorted_dist = dist.argsort()
    class_count = defaultdict(int)
    for i in range(k):
        vote_label = labels[sorted_dist[i]]
        class_count[vote_label] += 1
    max_type = -1
    max_count = -1
    for key, value in class_count.items():
        if value > max_count:
            max_count = value
            max_type = key
    return max_type

def mnist():
    origin_x_train, origin_y_train, origin_x_test, origin_y_test = load_mnist_data("mnist_train.csv","mnist_test.csv")
    # data visualization
    row = int(randrange(1, 9))

    print("Display an random image of number:", origin_y_train[row])

    plt.imshow(origin_x_train[row].reshape((28, 28)))
    plt.show()

    # Display partical images from dataset
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    rows = 4
    for y, cls in enumerate(classes):
        idxs = np.nonzero([i == y for i in origin_y_train])
        idxs = np.random.choice(idxs[0], rows)
        for i, idx in enumerate(idxs):
            plt_idx = i * len(classes) + y + 1
            plt.subplot(rows, len(classes), plt_idx)
            plt.imshow(origin_x_train[idx].reshape((28, 28)))
            plt.axis("off")
            if i == 0:
                plt.title(cls)
    plt.show()

    # print(list(origin_y_test))
    mnist_x_train = np.array(list(origin_x_train))
    mnist_y_train = np.array(list(origin_y_train))
    mnist_x_test = list(origin_x_test)
    standard_label = list(origin_y_test)
    success = 0
    failed = 0
    print("mnist test cases number:",len(mnist_x_test))
    for i in range(len(mnist_x_test)):
        inx = np.array(mnist_x_test[i])
        res = knn(inx, mnist_x_train, mnist_y_train, 5)
        if i % 250 == 0:
            print("mnist case:#%5d  " % (i), standard_label[i], "---", res)
        if standard_label[i] != res:
            failed += 1
        else:
            success += 1
    return i+1,success,failed


# pima
def load_pima_data(train_name,test_name):
    train = pd.read_csv(train_name)
    print("training data:", train.shape)
    x_train = train.values[:,:8]
    y_train = train.values[:,8]

    test = pd.read_csv(test_name)
    print("test data:", test.shape)
    x_test = test.values[:, :8]
    y_test = test.values[:, 8]
    return x_train, y_train, x_test, y_test

def pima():
    pima = pd.read_csv("pima_indians_diabetes.csv")
    print(pima.head())
    print(pima.shape)
    print(pima.describe())
    # histogram
    pima.hist(figsize=(16, 14))
    # scatter diagram
    sns.pairplot(pima, hue="Outcome")


    x_train, y_train, x_test, y_test = load_pima_data("pima_train.csv","pima_test.csv")
    pima_x_train = np.array(list(x_train))
    pima_y_train = np.array(list(y_train))
    pima_x_test = list(x_test)
    standard_label = list(y_test)
    success = 0
    failed = 0
    print("pima test cases number:",len(pima_x_test))
    for i in range(len(pima_x_test)):
        inx = np.array(pima_x_test[i])
        res = knn(inx, pima_x_train, pima_y_train, 5)
        if i % 10 == 0:
            print("pima case:#%2d  " % (i), standard_label[i], "---", res)
        if standard_label[i] != res:
            failed += 1
        else:
            success += 1
    return i+1,success,failed





# mnist
# mnist_tests_number,mnist_success_number,mnist_failed_number = mnist()
# print("%d mnist test cases: %d success, %d failed. Accurancy:=%.4f" % (mnist_tests_number,mnist_success_number,mnist_failed_number, mnist_success_number / mnist_tests_number))

# pima
pima_tests_number, pima_success_number, pima_failed_number = pima()
print("%d pima test cases: %d success, %d failed. Accurancy:=%.4f" % (pima_tests_number, pima_success_number, pima_failed_number, pima_success_number / pima_tests_number))


#
# def get_mnist_test_data(dataset_name):
#     data = pd.read_csv(dataset_name)
#     # select data except labels
#     images = data.iloc[:,1:].values
#     # flatten label
#     labels = data.iloc[:,:1].values.ravel()
#     images = np.multiply(images,1.0/255.0)
#     images = images.reshape(images.shape[0],1,28,28)
#     print(images)
#     return images,labels
