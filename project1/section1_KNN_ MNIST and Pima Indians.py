import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from random import randrange
from collections import defaultdict
import time
import datetime

def print_time():
    # t = time.time()
    # res = time.ctime(t)
    res = datetime.datetime.now()
    print(res)

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
def mnist_specific(d,k_scale):
    origin_x_train, origin_y_train, origin_x_test, origin_y_test = load_mnist_data("mnist_train.csv", "mnist_test.csv")
    mnist_x_train = np.array(list(origin_x_train))
    mnist_y_train = np.array(list(origin_y_train))
    mnist_x_test = list(origin_x_test)
    standard_label = list(origin_y_test)

    res = [0 for _ in range(k_scale)]
    total = 0
    for i in range(len(mnist_x_test)):
        # if i > 200:
        #     break
        if standard_label[i] == d:
            total += 1
            inx = np.array(mnist_x_test[i])

            dist = (((mnist_x_train - inx) ** 2).sum(1)) ** 0.5
            sorted_dist = dist.argsort()
            class_count = defaultdict(int)
            for k in range(1,k_scale):
                for j in range(k):
                    vote_label = mnist_y_train[sorted_dist[j]]
                    class_count[vote_label] += 1
                max_type = -1
                max_count = -1
                for key, value in class_count.items():
                    if value > max_count:
                        max_count = value
                        max_type = key
                if max_type == standard_label[i]:
                    res[k] += 1
    temp_y = [_ for _ in range(k_scale)]
    for i in range(k_scale):
        res[i] /= total
    plt.plot(temp_y[1:],res[1:],'ro-')
    plt.title('digit = %d'%d)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()



def mnist(k):
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
    test_predictions = []
    print("mnist test cases number:",len(mnist_x_test))
    print("time-----------------------")
    print_time()
    m_time_1 = time.time()
    confusion = [[0 for _ in range(10)] for __ in range(10)]
    for i in range(len(mnist_x_test)):
        inx = np.array(mnist_x_test[i])
        res = knn(inx, mnist_x_train, mnist_y_train, k)
        test_predictions.append(res)
        confusion[standard_label[i]][res] += 1
        # if i > 100:
        #     break
        # if i % 250 == 0:
        #     print("mnist case:#%5d  " % (i), standard_label[i], "---", res)
        if standard_label[i] != res:
            failed += 1
        else:
            success += 1
    print_time()
    m_time_2 = time.time()
    print("MNIST Running time is: %.6f seconds" % (m_time_2 - m_time_1))
    return i+1,success,failed,confusion

def confusion_matrix(matrix):
    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(matrix, annot=True, cmap='Blues')

    plt.ylim(0, 10)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

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
    print("time-----------------")

    confusion = [[0 for _ in range(2)] for __ in range(2)]

    print_time()
    p_time_1 = time.time()
    for i in range(len(pima_x_test)):
        inx = np.array(pima_x_test[i])
        res = knn(inx, pima_x_train, pima_y_train, 5)
        # if i % 10 == 0:
        #     print("pima case:#%2d  " % (i), standard_label[i], "---", res)
        if standard_label[i] > 0:
            if res > 0:
                confusion[1][1] += 1
            else:
                confusion[1][0] += 1
        else:
            if res > 0:
                confusion[0][1] += 1
            else:
                confusion[0][0] += 1
        if standard_label[i] != res:
            failed += 1
        else:
            success += 1
    print_time()
    p_time_2 = time.time()
    print("PIMA Running time is: %.6f seconds"%(p_time_2-p_time_1))
    return i+1,success,failed,confusion

def confusion_matrix_pima(matrix):
    figure = plt.figure(figsize=(2, 2))
    sns.heatmap(matrix, annot=True, cmap='Blues')

    plt.ylim(2)
    # plt.title("Euclidean Distance")
    # plt.title("Manhattan Distance")
    plt.title("Chebyshev Distance")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


# mnist
mnist_tests_number,mnist_success_number,mnist_failed_number,confusion = mnist(5)
# confusion_matrix(confusion)
# print("%d mnist test cases: %d success, %d failed. Accurancy:=%.4f" % (mnist_tests_number,mnist_success_number,mnist_failed_number, mnist_success_number / mnist_tests_number))
# mnist_specific(7,100)



# pima
pima_tests_number, pima_success_number, pima_failed_number,confusion = pima()
confusion_matrix_pima(confusion)
print("%d pima test cases: %d success, %d failed. Accurancy:=%.4f" % (pima_tests_number, pima_success_number, pima_failed_number, pima_success_number / pima_tests_number))


