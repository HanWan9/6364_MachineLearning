import csv
import collections
from numpy import *
import random
import matplotlib.pyplot as plt
from itertools import chain, combinations

def shuffle_data(data):
    random.shuffle(data)

def pre_process_data(file_name: str) -> [dict]:
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # the last read is [], so we need to test if row is []
        data_set = [row[0].split(';') for row in reader]

    keys = data_set[0]

    names = set(['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])
    res = collections.defaultdict(set)
    for data in data_set[1:]:
        for i in range(len(data)):
            if keys[i] not in names:
                continue
            res[keys[i]].add(data[i])
    for k, v in res.items():
        res[k] = sorted(list(v))
    str2num = collections.defaultdict(dict)
    for k, v in res.items():
        for i in range(len(v)):
            str2num[k][v[i]] = i
    print(str2num)
    return_data = data_set[1:]
    for i in range(len(return_data)):
        for j in range(len(return_data[i])):
            if keys[j] in names:
                return_data[i][j] = str2num[keys[j]][return_data[i][j]]
            else:
                if return_data[i][j].startswith('"'):
                    return_data[i][j] = return_data[i][j][1:-1]
                return_data[i][j] = int(return_data[i][j])
    # ['0:school', '1:sex', '2:age', '3:address', '4:famsize', '5:Pstatus', '6:Medu', '7:Fedu', '8:Mjob', '9:Fjob', '10:reason',
    # '11:guardian', '12:traveltime', '13:studytime', '14:failures', '15:schoolsup', '16:famsup', '17:paid', '18:activities',
    # '19:nursery', '20:higher', '21:internet', '22:romantic', '23:famrel', '24:freetime', '25:goout', '26:Dalc', '27:Walc',
    # '28:health', '29:absences', '30:G1', '31:G2', '32:G3']
    #return return_data
    rand_group = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    #rand_group = [3,7,9,12,25,27,29,]
    #rand_group = [15,17, 20, 21, 22, 25, 26, 30]
    #rand_group = [3,13,15, 17] # address + studyTime + schoolsup + paid ~ 14.84
    #rand_group = [3, 13,15, 1] # address + studyTime + schoolsup + famsup ~ 1
    #rand_group = [3,13,15] # address + studyTime + schoolsup ~ 11.87
    #rand_group = [3, 8, 13] #address+studytime+Mjob ~ 2490.66
    #rand_group = [3, 13] # address+studytime~1
    #rand_group = [13] # studytime~2
    #rand_group = list(range(33)))
    #rand_group = list(range(18)) # diff~5
    rlt = []
    for data in return_data:
        temp = []
        for i in rand_group:
            temp.append(data[i])
        rlt.append(temp)
    #shuffle_data(rlt)
    return rlt

def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))

def cal_G3(all_data):
    num_data = len(all_data)
    row_num = num_data * 10 // 9
    training_set = all_data[:row_num]
    testing_set = all_data[:row_num]
    w = cal_w(training_set)

    x = [row[:-1] for row in testing_set]
    y = [row[-1] for row in testing_set]

    sum = 0
    for i in range(len(x)):
        sum += abs(predict(w, mat(x[i])) - y[i])
    return [sum, w]

def find_best(file_name):
    data_set = pre_process_data(file_name)
    n = len(data_set[0])
    iterable = list(range(n))
    rand_groups = list(powerset(iterable))[1:]
    min_diff = float('inf')
    best_w = None
    for rand_group in rand_groups:
        rlt = []
        for data in data_set:
            temp = []
            for i in rand_group:
                temp.append(data[i])
            rlt.append(temp)
        diff, w = cal_G3(rlt)
        if diff < min_diff:
            min_diff = diff
            best_w = w
    print(min_diff)
    print(best_w)


def cal_w(training_set):
    x = [[1] + row[:-1] for row in training_set]
    y = [[row[-1]] for row in training_set]
    x_matrix = mat(x)
    y_matrix = mat(y)
    w = (x_matrix.T * x_matrix).I * x_matrix.T * y_matrix
    return w

def predict(w, data):
    return sum((w * data).tolist())


def linear_regression(file_name):
    all_data = pre_process_data(file_name)
    num_data = len(all_data)
    row_num = num_data * 9 // 10
    training_set = all_data[num_data * 1 // 10:row_num]
    testing_set = all_data[:num_data * 1 // 10] + all_data[row_num:]
    w = cal_w(training_set)

    x = [row[:-1] for row in testing_set]
    y = [row[-1] for row in testing_set]

    draw_data = [[i] + [training_set[i][-1]] for i in range(len(training_set))]
    draw(draw_data)
    sum = 0
    for i in range(len(x)):
        sum += (predict(w, mat(x[i])) - y[i]) ** 2
        print(predict(w, mat(x[i])))
    print(sum / len(x))


def draw(data):
    N = len(data)
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    area = [10] * N
    plt.scatter(x, y, s=area, c='green', alpha=0.6)
    plt.show()


rlt = linear_regression("student-por.csv")
find_best("student-por.csv")
# ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',        'Fjob',     'reason',    'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',  'activities', 'nursery',  'higher',  'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',  'G1',    'G2',  'G3']
# ['GP',     '"F"', '18',   '"U"',     '"GT3"',   '"A"',     '4',    '4',    '"at_home"', '"teacher"', '"course"', '"mother"',  '2',           '2',        '0',         '"yes"',     '"no"',   '"no"', '"no"',        '"yes"',    '"yes"',  '"no"',      '"no"',    '4',      '3',         '4',    '1',     '1',    '3',      '4',        '"0"',  '"11"', '11']
