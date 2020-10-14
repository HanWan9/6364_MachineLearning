import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


#
# data_x,data_y,data_y_ = load_data("student-por.csv")
# theta = np.zeros([1,3])
# xx = data_x[:, [28, 29]]
# t = np.ones((len(xx),1))
# xxx = np.hstack((t,xx))
# xxx = np.array(xxx,dtype=int)
# res = []
# for i in range(17):
#     yyy = data_y_[:,[i]]
#     data = np.hstack((xxx,yyy))
#     # print(xxx[:5])
#     # print(yyy[0,:5])
#     # print(xxx.shape,yyy.shape,theta.shape)
#     # print(cost(xxx,yyy,theta))
#     predictions = predict(xxx,theta)
#     res.append(predictions)
#     print(len(predictions),predictions)
#     # print(type(predictions))
#     print(sum(predictions))
#     correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a,b)in zip(predictions,yyy)]
#     accuracy = (sum(map(int,correct)) / len(correct))
#     print("accuracy = {0}%".format(accuracy))
# print(len(res))
# correct = 0
# for i in range(len(res[0])):
#     for j in range(len(res)):
#         if res[j][i] == 1 and data_y_[i,j] == 1:
#             correct += 1
#             break
#
# print("correct=%d, total=%d, accuract=%.4f"%(correct,i+1,correct/(i+1)))



def loadcsv():
    tmp = np.loadtxt("student-por.csv", dtype=np.str, delimiter=';')
    data = tmp[1:]
    # 分类标准
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][0] == '"':
                data[i][j] = data[i][j][1:-1]
    id0 = ['no']
    id1 = ['GP', 'yes', 'F', 'U', 'LE3', 'T', 'teacher', 'home', 'mother']
    id2 = ['MS', 'M', 'R', 'GT3', 'A', 'health', 'reputation', 'father']
    id3 = ['services', 'course']
    id4 = ['at_home']
    id5 = ['other']
    for each in id0:
        index = np.where(data == each)
        data[index] = 0
    for each in id1:
        index = np.where(data == each)
        data[index] = 1
    for each in id2:
        index = np.where(data == each)
        data[index] = 2
    for each in id3:
        index = np.where(data == each)
        data[index] = 3
    for each in id4:
        index = np.where(data == each)
        data[index] = 4
    for each in id5:
        index = np.where(data == each)
        data[index] = 5
    data = data.astype(np.int)
    return data

    # The column you could choose

def get_fea_lab(data):
    # X = data[:, 30: 32]
    X = data[:, :30]
    # X = data[:, [12,13,24]]
    y = data[:, -1]
    X = np.mat(X)
    y = np.mat(y)
    return X, y


# # Define the cost function:
# def computeCost(data, theta, i):
#     X, y = get_fea_lab(data)
#     inner = np.dot(X, theta.T) - y.T
#     inner = np.power(inner, 2)
#     return (float(inner[i] / 2))


# Define the stochastic gradient descent function:
def stochastic_gradient_descent(data, theta, alpha, epoch):
    X0, y0 = get_fea_lab(data)
    temp = np.mat(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(len(X0))
    avg_cost = np.zeros(epoch)

    for k in range(epoch):
        np.random.shuffle(data)
        X, y = get_fea_lab(data)

        for i in range(len(X)):
            error = X[i] * theta.T - y[0, i]
            cost[i] = float(error) ** 2 / 2
            for j in range(parameters):
                temp[0, j] = theta[0, j] - alpha * error * X[i, j]
            theta = temp
        avg_cost[k] = np.average(cost)

    return theta, avg_cost

# ['0:school', '1:sex', '2:age', '3:address', '4:famsize', '5:Pstatus', '6:Medu', '7:Fedu', '8:Mjob', '9:Fjob', '10:reason',
    # '11:guardian', '12:traveltime', '13:studytime', '14:failures', '15:schoolsup', '16:famsup', '17:paid', '18:activities',
    # '19:nursery', '20:higher', '21:internet', '22:romantic', '23:famrel', '24:freetime', '25:goout', '26:Dalc', '27:Walc',
    # '28:health', '29:absences', '30:G1', '31:G2', '32:G3']
data = loadcsv()
# num = len(data[0])-1
num = 30
alpha = 0.0001
epoch = 100
theta = np.mat(np.zeros((1, num)))
g, avg_cost = stochastic_gradient_descent(data, theta, alpha, epoch)
print(avg_cost[-1])
plt.plot(avg_cost)
plt.title('Mean sqrt Error: %.6f for All features except G1,G2'%avg_cost[-1])
plt.show()

# ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',        'Fjob',     'reason',    'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',  'activities', 'nursery',  'higher',  'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',  'G1',    'G2',  'G3']
# ['GP',     '"F"', '18',   '"U"',     '"GT3"',   '"A"',     '4',    '4',    '"at_home"', '"teacher"', '"course"', '"mother"',  '2',           '2',        '0',         '"yes"',     '"no"',   '"no"', '"no"',        '"yes"',    '"yes"',  '"no"',      '"no"',    '4',      '3',         '4',    '1',     '1',    '3',      '4',        '"0"',  '"11"', '11']

