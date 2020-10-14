import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from random import randrange
from collections import defaultdict
import time
import datetime

def split_data(data,test_size):
    values = data.values
    np.random.shuffle(values)
    l = len(values)
    train_number = int(l*(1-test_size))
    return values[:train_number],values[train_number:]

def load_mnist_data(data_name):
    data = pd.read_csv(data_name,header=None)
    # print("data scale:", data.shape)
    train_data, test_data = split_data(data, 0.23)
    print("train_data scale:", train_data.shape)
    print("test_data scale:", test_data.shape)
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    y_train = np.where(y_train == 1, 1, -1)
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    y_test = np.where(y_test == 1, 1, -1)
    return x_train, y_train, x_test, y_test

def knn(inx,dataset,labels,k):
    # Euclidean
    dist = (((dataset-inx)**2).sum(1))**0.5
    # Manhattan
    # dist = ((abs(dataset - inx)).sum(1))
    # Chebyshev
    # dist = (np.abs(dataset.max()-inx))
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

class Perceptron():

    def __init__(self, learning_rate=  0.005, num_iter=20, random_state=1):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.random_state = random_state
        self.confidence = []

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.num_iter):
            errors = 0
            for x_i, target in zip(X, y):
                # 分类正确不更新，分类错误更新权重
                update = self.learning_rate * (target - self.predict(x_i))
                self.w_[1:] += update * x_i
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def predict_input(self, X):
        """计算预测值"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        """得出sign(预测值)即分类结果"""
        return np.where(self.predict_input(X) >= 0.0, 1, -1)

def confusion_matrix(matrix,method,accuracy,case_number):
    figure = plt.figure(figsize=(4, 4))
    sns.heatmap(matrix, annot=True, cmap='Blues')
    plt.ylim(2)
    plt.title("test case #%d\n"%case_number+method+" Accuracy = %.4f"%accuracy)
    plt.xlabel('Predicted Positive    Predicted Negative')
    plt.ylabel('Actual Positive          Actual Negative')
    plt.show()

def execute(case_number = 1):
    train_x,train_y,test_x,test_y = load_mnist_data("wine.data")
    standard_label = list(test_y)
    X = list(test_x)
    total = len(standard_label)
    # knn
    confusion_knn = [[0 for _ in range(2)] for __ in range(2)]
    success_knn = 0
    for i in range(len(X)):
        x_i = test_x[i]
        res = knn(x_i, train_x, train_y, 5)
        if standard_label[i] == res:
            success_knn += 1
        if standard_label[i] > 0:
            if res > 0:
                confusion_knn[1][0] += 1
            else:
                confusion_knn[1][1] += 1
        else:
            if res > 0:
                confusion_knn[0][0] += 1
            else:
                confusion_knn[0][1] += 1
    accuracy_knn = success_knn / total
    print(success_knn,"Accuracy_knn = %.4f"%accuracy_knn)

    # perceptron
    confusion_perceptron = [[0 for _ in range(2)] for __ in range(2)]
    wine = Perceptron(learning_rate= 0.00005, num_iter=25)
    wine.fit(train_x, train_y)
    success_perceptron = 0
    for i in range(len(X)):
        res = wine.predict(test_x[i])
        if res == standard_label[i]:
            success_perceptron += 1
        if standard_label[i] > 0:
            if res > 0:
                confusion_perceptron[1][0] += 1
            else:
                confusion_perceptron[1][1] += 1
        else:
            if res > 0:
                confusion_perceptron[0][0] += 1
            else:
                confusion_perceptron[0][1] += 1
    accuracy_perceptron = success_perceptron / total
    print(success_perceptron,"Accuracy_Per = %.4f"%accuracy_perceptron)

    # confusion matrix
    confusion_matrix(confusion_knn,"KNN",accuracy_knn,case_number)
    confusion_matrix(confusion_perceptron,"Perceptron",accuracy_perceptron,case_number)

    # return accuracy_knn,accuracy_perceptron

execute()
# knns = []
# percs = []
# times = 30
# for i in range(times):
#     knn, perc = execute(i)
#     knns.append(knn)
#     percs.append(perc)
# max_k = max(knns)
# max_p = max(percs)
# min_k = min(knns)
# min_p = min(percs)
# same = 0
# knn_better = 0
# perc_better = 0
# for i in range(times):
#     if knns[i] == percs[i]:
#         same += 1
#     elif knns[i] > percs[i]:
#         knn_better += 1
#     else:
#         perc_better += 1
#
# print("The number of test times: %d"%times)
# print("Max accuracy: KNN = %.3f; Perceptron = %.3f"%(max_k,max_p))
# print("Min accuracy: KNN = %.3f; Perceptron = %.3f"%(min_k,min_p))
# print("KNN        performs better times: %d"%knn_better)
# print("Perceptron performs better times: %d"%perc_better)
# print("Same times: %d"%same)
