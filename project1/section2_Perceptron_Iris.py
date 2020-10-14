import numpy as np
import pandas as pd
from numpy import *
from pandas import *
import matplotlib.pyplot as plt
from numpy.random import seed
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc")


# class Perceptron():
#     def __init__(self, learning_rate, n_iters):
#         self.iters = n_iters
#         self.eta = learning_rate
#         self.loss_tolerance = 20
#
#     # 模型训练
#     def fit(self, X, Y):
#         n_sample, n_feature = X.shape
#         rnd_val = 1/np.sqrt(n_feature)
#         rng = np.random.default_rng()
#
#         # 均匀初始化权重
#         self.w = rng.uniform(-rnd_val, rnd_val, size=n_feature)
#         self.b = 0
#
#         # 随机梯度迭代过程
#         iter = 0 #迭代的次数
#         prev_loss = 0 #上一轮迭代的损失
#
#         while True:
#             cur_loss = 0 #当前轮的损失
#             wrong_classify = 0 #误分类样本个数
#
#             for i in range(n_sample):
#                 y_pred = np.dot(self.w, X[i])+self.b
#                 cur_loss += -Y[i] * y_pred
#
#                 # 对误分类样本进行参数更新
#                 if Y[i] * y_pred < 0:
#                     # self.w += self.eta*Y[i]*X[i]
#                     np.add(self.w, self.eta*Y[i]*X[i], out = self.w, casting="unsafe")
#                     self.b += self.eta*Y[i]
#             iter += 1
#             # loss_diff = abs(cur_loss - prev_loss)
#             prev_loss = cur_loss
#
#             # 模型停止训练的条件：
#             # 1. 训练的epoch数达到指定的epoch数
#             # 2. 当前epoch数与上一轮epoch损失之差小雨给定阈值
#             # 3. 训练过程中不再出现误分类样本
#             # if iter > self.iters or loss_diff < self.loss_tolerance or wrong_classify == 0:
#             if iter > self.iters or wrong_classify == 0:
#                 break
#
#
#
#
#
#     def predict(self,xi):
#         res = []
#         for x in xi:
#             y_predict = np.dot(self.w, x) + self.b
#             res.append(1 if y_predict >= 0 else -1)
#         return res

class Perceptron():

    def __init__(self, learning_rate=  0.005, num_iter=20, random_state=1):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.random_state = random_state
        self.confidence = []

    def fit(self, X, y):
        """初始化并更新权重"""
        # 通过标准差为0.01的正态分布初始化权重
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.errors_ = []

        # 循环遍历更新权重直至算法收敛
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

    def confidence_metric(self,X):
        for x_i in X:
            predict = self.predict_input(x_i)
            self.confidence.append(predict*100)
        return self.confidence


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # 构造颜色映射关系
    marker_list = ['o', 'x', 's']
    color_list = ['r', 'b', 'g']
    cmap = ListedColormap(color_list[:len(np.unique(y))])

    # 构造网格采样点并使用算法训练阵列中每个元素
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 第0列的范围
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, 666)  # 横轴采样多少个点
    t2 = np.linspace(x2_min, x2_max, 666)  # 纵轴采样多少个点
#     t1 = np.arange(x1_min, x1_max, resolution)
#     t2 = np.arange(x2_min, x2_max, resolution)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
#     y_hat = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T) # 预测值
    y_hat = classifier.predict(np.stack((x1.flat, x2.flat), axis=1))  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同

    # 通过网格采样点画出等高线图
    plt.contourf(x1, x2, y_hat, alpha=0.2, cmap=cmap)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())

    for ind, clas in enumerate(np.unique(y)):
        plt.scatter(X[y == clas, 0], X[y == clas, 1], alpha=0.8, s=50,
                    c=color_list[ind], marker=marker_list[ind], label=clas)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    res = []
    for i in z:
        t = 1-i
        res.append(t if t<1 else t-1)
    return res

df = pd.read_csv('iris.data', header=None)

# 取出前100行的第五列即生成标记向量
y = df.iloc[:, 4].values
y = np.where(y == 'Iris-virginica', 1, -1)

# 取出前100行的第一列和第三列的特征即生成特征向量

X = df.iloc[:, [2,3]].values

plt.scatter(X[:50, 0], X[:50, 1], color='r', s=50, marker='x', label='Iris-setosa')
plt.scatter(X[50:, 0], X[50:, 1], color='b',
            s=50, marker='o', label='Iris-virginica')
plt.xlabel('Petal length（cm）', fontproperties=font)
plt.ylabel('Petal width（cm）', fontproperties=font)
plt.legend(prop=font)
plt.show()



# learning rate = 0.00005, 0.001, 0.005
perceptron = Perceptron(learning_rate= 0.00005, num_iter=25)
perceptron.fit(X, y)

# confidence metric
# z = np.arange(-10,10,0.1)
predictions = perceptron.confidence_metric(X)
z = np.array(predictions)
sigmoid = sigmoid(z)
print(sigmoid)
plt.plot(range(1, len(z) + 1),sigmoid)
plt.scatter(range(1, len(z) + 1),sigmoid)

#画一条竖直线，如果不设定x的值，则默认是0
# plt.axvline(x=0, color='k')
# plt.axhspan(0.0, 1.0,facecolor='0.7',alpha=0.4)
# 画一条水平线，如果不设定y的值，则默认是0
# plt.axhline(y=1, ls='dotted', color='0.4')
# plt.axhline(y=0, ls='dotted', color='0.4')
plt.axhline(y=0.5, ls='dotted', color='k')
# plt.ylim(-0.1,1.1)
#确定y轴的坐标
# plt.yticks([0.0, 0.5, 1.0])
plt.title('Confidence Metrics with Learning Rate 0.00005')
# plt.ylabel('Confidence')
plt.xlabel('Samples')
ax = plt.gca()
ax.grid(True)
plt.show()


plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel('Number of iterations', fontproperties=font)
plt.ylabel('Number of updates', fontproperties=font)
plt.title('Learning Rate = 0.00005')
plt.show()




plot_decision_regions(X, y, classifier=perceptron)
plt.xlabel('Petal length（cm）', fontproperties=font)
plt.ylabel('Petal width（cm）', fontproperties=font)
plt.legend(prop=font)
plt.show()
#
# def split_data(data,test_size):
#     values = data.values
#     np.random.shuffle(values)
#     l = len(values)
#     train_number = int(l*(1-test_size))
#     return values[:train_number],values[train_number:]
#
# def load_data(data_name, test_size):
#     names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#     data = pd.read_csv(data_name, header=None, names=names)
#     print("data scale:", data.shape)
#
#     data_train, data_test = split_data(data,test_size)
#
#     data_train_x = data_train[:, 2:4]
#     data_train_y = data_train[:, 4]
#     for i in range(len(data_train_y)):
#         if data_train_y[i] == "Iris-virginica":
#             data_train_y[i] = 1
#         elif data_train_y[i] == "Iris-setosa":
#             data_train_y[i] = -1
#     data_test_x = data_test[:, 2:4]
#     data_test_y = data_test[:, 4]
#     for i in range(len(data_test_y)):
#         if data_test_y[i] == "Iris-virginica":
#             data_test_y[i] = 1
#         elif data_test_y[i] == "Iris-setosa":
#             data_test_y[i] = -1
#     return data,data_train_x,data_train_y,data_test_x,data_test_y
#
# def accuracy(y_true, y_pred):
#     accuracy = np.sum(y_true == y_pred) / len(y_true)
#     return accuracy
#
# def plot_decision_regions(x, y, classifier, resolution=0.2):
#     """
#     二维数据集决策边界可视化
#     :parameter
#     -----------------------------
#     :param self: 将鸢尾花花萼长度、花瓣长度进行可视化及分类
#     :param x: list 被分类的样本
#     :param y: list 样本对应的真实分类
#     :param classifier: method  分类器：感知器
#     :param resolution:
#     :return:
#     -----------------------------
#     """
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     # y去重之后的种类
#     listedColormap = ListedColormap(colors[:len(np.unique(y))])
#
#     # 花萼长度最小值-1，最大值+1
#     x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
#     # 花瓣长度最小值-1，最大值+1
#     x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
#
#     # 将最大值，最小值向量生成二维数组xx1,xx2
#     # np.arange(x1_min, x1_max, resolution)  最小值最大值中间，步长为resolution
#     new_x1 = np.arange(x1_min, x1_max, resolution)
#     new_x2 = np.arange(x2_min, x2_max, resolution)
#     xx1, xx2 = np.meshgrid(new_x1, new_x2)
#
#     # 预测值
#     # z = classifier.predict([xx1, xx2])
#     z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     z = z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, z, alpha=0.4, camp=listedColormap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#
#     for idx, c1 in enumerate(np.unique(y)):
#         plt.scatter(x=x[y == c1, 0], y=x[y == c1, 1], alpha=0.8, c=listedColormap(idx), marker=markers[idx], label=c1)
#
#
# def plot_table(row, col, vals):
#     data = {}
#     for i in range(len(col)):
#         data[col[i]] = vals[i]
#     df = pd.DataFrame(data, index=row)
#     print(df)
#     df.plot(kind='bar', grid=True, colormap='Blues_r', stacked=True, edgecolor='black', rot=0)
#
#
#
#
# # 1. What happens when the learning rate is 0.00005, 0.001, and 0.005?
# # 2. Does the algorithm converge?
# #    Plot the classification accuracy for each learning rate from
# #    1 to 20 training epochs.
# # 3. Come up with a confidence metric in your classification.
# #    (For example come up with an activation function that might correspond to confidence.)
# #    Create a scatter plot for confidence vs classification result for all instances with learning rate 0.00005.
# # 4. Is this data set linearly separable?
# #    Justify your answer with a scatterplot. Explain why and how you created this scatterplot.
#
#
# data,x_train,y_train,x_test,y_test = load_data("iris.data",test_size = 0.25)
#
# # 0到100行，第5列
# y = data.iloc[0:100, 4].values
# # 将target值转数字化 Iris-setosa为-1，否则值为1
# y = np.where(y == "Iris-setosa", -1, 1)
# # 取出0到100行，第2，第3列的值
# x = data.iloc[0:100, [2, 3]].values
#
# """ 鸢尾花散点图 """
# # scatter绘制点图
# plt.scatter(x[0:50, 0], x[0:50, 1], color="red", marker="o", label="setosa")
# plt.scatter(x[50:100, 0], x[50:100, 1], color="blue", marker="x", label="versicolor")
# # 防止中文乱码 下面分别是windows系统，mac系统解决中文乱码方案
# # zhfont1 = mat.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
# plt.title("scatter")
# plt.xlabel(u"petal length")
# plt.ylabel(u"petal width")
# plt.legend(loc="upper left")
# plt.show()
#
#
#
# learning_rates = [0.00005, 0.001, 0.005]
# iters = [_+1 for _ in range(20)]
# if_test = False
# accuracies = []
# if if_test:
#     p = Perceptron(learning_rate=0.001, n_iters=10)
#     p.fit(x_train,y_train)
#     predictions = p.predict(x_test)
#     print("预测值:", predictions)
#     print("真实值:", y_test)
#     print("Perceptron classificaiton accuracy:", accuracy(y_test,predictions))
#
#
#     plot_decision_regions(x, y, classifier=p)
#     plt.title("section_2:Perceptron-Iris")
#     plt.xlabel("")
#     plt.ylabel("")
#     plt.legend(loc="upper left")
#     plt.show()
#
#
# else:
#     for lr in learning_rates:
#         path = []
#         for eta in iters:
#             p = Perceptron(lr,eta)
#             p.fit(x_train, y_train)
#             predictions = p.predict(x_test)
#             path.append(accuracy(y_test,predictions))
#         accuracies.append(path)
#
#
#         print(lr,sum(path)/20)
#         plt.plot(iters,path)
#         plt.show()
#
#
#     plot_table(iters,learning_rates,accuracies)



















