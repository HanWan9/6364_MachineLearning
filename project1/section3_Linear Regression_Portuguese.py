import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
import numpy.random
import time

# load data
def split_data(data,test_size):
    values = data.values
    np.random.shuffle(values)
    l = len(values)
    train_number = int(l*(1-test_size))
    return values[:train_number],values[train_number:]

def translate_y(y):
    a = np.mat(y).ravel()
    b = a.tolist()
    b = b[0][:]
    c = set(b)
    l = len(b)
    row = {}
    i = 0
    for label in c:
        row[label] = i
        i += 1
    res = np.zeros((l,17))
    for j in range(len(y)):
        label = int(y[j])
        idx = row[label]
        res[j,idx] = 1
    return res

# def load_data(data_name,test_size):
#     data = pd.read_csv(data_name,sep=";")
#     print(data.head())
#     print("data scale:", data.shape)
#     features = list(data.columns)
#     features_x = features[:30]+features[32:33]
#
#     data_train,data_test = split_data(data,test_size)
#     # select features
#     train_x = data_train[:, :30]
#     train_y = data_train[:, 32:33]
#     test_x = data_test[:, :30]
#     test_y = data_test[:, 32:33]
#     # print(train_y.shape,test_y.shape)
#
#     train_y_ = translate_y(train_y)
#     test_y_ = translate_y(test_y)
#     # print(train_y_.shape, test_y_.shape)
#
#     feature_y = np.mat(data).ravel().tolist()[0][:]
#
#     # print(train_y_)
#     # print("*********")
#     # print(test_y_)
#     xxx = data_train[:,[28,29]]
#     print(xxx)
#     # xxx = np.hstack((xxx,train_y_[0]))
#
#     return data_train,data_test,features_x,feature_y,train_x,train_y_,test_x,test_y_,test_y, xxx

def load_data(data_name):
    data = pd.read_csv(data_name,sep=";")
    print(data.head())
    print("data scale:", data.shape)


    # select features
    data_x = data.values[:, :30]
    data_y = data.values[:, 32:33]
    data_y_ = translate_y(data_y)

    return data_x,data_y,data_y_



def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def model(x, theta):
    return sigmoid(np.dot(x,theta.T))
# 损失
def cost(x,y,theta):
    left = np.multiply(-y, np.log(model(x,theta)))
    right = np.multiply(1-y, np.log(1-model(x,theta)))
    return np.sum(left-right) / (len(x))
# 偏导
def gradient(x,y,theta):
    grad = np.zeros(theta.shape)
    error = (model(x,theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, x[:,j])
        grad[0,j] = np.sum(term) / len(x)
    return grad

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2
def stop_criterion(type, value, threshold):
#     设定三种不同的停止策略
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold
# 洗牌
def shuffle_data(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    x = data[:, 0:cols-1]
    y = data[:,cols-1:]
    return x,y

def descent(data,theta, batchsize, stoptype, thresh, alpha):
    init_time = time.time()
    i = 0 # iters
    k = 0 # batch
    x,y = shuffle_data(data)
    grad = np.zeros(theta,theta)
    costs = [cost(x,y,theta)]
    while True:
        grad = gradient(x[k:k+batchsize],y[k:k+batchsize], theta)
        k += batchsize
        if k >= 518:
            k = 0
            x,y = shuffle_data(data)
        theta = theta - alpha*grad
        costs.append(cost(x,y,theta))
        i += 1

        if stoptype == STOP_ITER:
            value = i
        elif stoptype == STOP_COST:
            value = costs
        elif stoptype == STOP_GRAD:
            value = grad
        if stop_criterion(stoptype,value,thresh):
            break
    return theta, i-1, costs, grad, time.time() - init_time

# def runexpe(data,theta,batchsize,stoptype,thresh,alpha)

def predict(x,theta):
    return [1 if x > 0.5 else 0 for x in model(x,theta)]




# data_train,data_test,feature_x,feature_y,train_x,train_y,test_x,test_y,standard_y,xxx = load_data("student-por.csv",test_size=0.2)
# print("train_x scale:",train_x.shape)
# print("train_y scale:",train_y.shape)
# print("test_x scale:",test_x.shape)
# print("test_y scale:",test_y.shape)
# print("theta scale:",theta.shape)



data_x,data_y,data_y_ = load_data("student-por.csv")
theta = np.zeros([1,3])
xx = data_x[:, [28, 29]]
t = np.ones((len(xx),1))
xxx = np.hstack((t,xx))
xxx = np.array(xxx,dtype=int)
res = []
for i in range(17):
    yyy = data_y_[:,[i]]
    data = np.hstack((xxx,yyy))
    # print(xxx[:5])
    # print(yyy[0,:5])
    # print(xxx.shape,yyy.shape,theta.shape)
    # print(cost(xxx,yyy,theta))
    predictions = predict(xxx,theta)
    res.append(predictions)
    print(len(predictions),predictions)
    # print(type(predictions))
    print(sum(predictions))
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a,b)in zip(predictions,yyy)]
    accuracy = (sum(map(int,correct)) / len(correct))
    print("accuracy = {0}%".format(accuracy))
print(len(res))
correct = 0
for i in range(len(res[0])):
    for j in range(len(res)):
        if res[j][i] == 1 and data_y_[i,j] == 1:
            correct += 1
            break

print("correct=%d, total=%d, accuract=%.4f"%(correct,i+1,correct/(i+1)))

