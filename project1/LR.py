import numpy as np
from matplotlib import pyplot as plt



def loadcsv():
    tmp = np.loadtxt("student-por.csv", dtype=np.str, delimiter=';')
    data = tmp[1:]
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


def get_fea_lab(data):
    X = data[:, 0:-1]
    y = data[:, -1]
    X = np.mat(X)
    y = np.mat(y)
    return X, y


# # 定义代价函数：
# def computeCost(data, theta, i):
#     X, y = get_fea_lab(data)
#     inner = np.dot(X, theta.T) - y.T
#     inner = np.power(inner, 2)
#     return (float(inner[i] / 2))


# 定义随机梯度下降函数：
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


if __name__ == '__main__':
    data = loadcsv()
    alpha = 0.001
    epoch = 10
    theta = np.mat(np.zeros((1, len(data[0])-1)))
    g, avg_cost = stochastic_gradient_descent(data, theta, alpha, epoch)
    print(theta)
    plt.plot(avg_cost)
    plt.show()

