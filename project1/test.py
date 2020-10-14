import numpy as np
# import pandas as pd
# from scipy import stats
import statsmodels.api as sm

# import matplotlib.pyplot as plt

'''
验证置信度、置信区间
1、首先创建一个3.25万个数字的numpy数组，65%的数据为1，其余数据为0
'''
love_soccer_prop = 0.65  # 实际足球爱好者比例
total_population = 325 * 10 ** 2  # 总人口

num_people_love_soccer = int(total_population * love_soccer_prop)
num_people_dont_love_soccer = int(total_population * (1 - love_soccer_prop))

people_love_soccer = np.ones(num_people_love_soccer)
people_dont_love_soccer = np.zeros(num_people_dont_love_soccer)

all_people = np.hstack([people_love_soccer, people_dont_love_soccer])

print("\n 取得总体平均值：")
print(np.mean(all_people))

'''
2、现在抽取几组容量为1000的样本，看看得到的百分比是多少
'''
print("\n 10次，随机抽取1000个样本，分别计算平均值：")
for i in range(10):
    aa = np.mean(np.random.choice(all_people, size=1000))
    print(aa)

'''
3、扩大抽样次数，看看得到的百分比是多少
应该是更接近65%了
'''
print("\n 进行10000次，随机抽取1000个样本，计算所有数据的平均值：")
values = []
for i in range(10000):
    sample = np.random.choice(all_people, size=1000)
    mean = np.mean(sample)
    values.append(mean)

print(np.mean(values))

'''
4、抽样100次，分别计算置信区间
这100个区间中，至少有95个区间包含上面计算出总体平均值
'''
print("\n 进行100次，随机抽取1000个样本，置信度95%，计算置信区间：")

for i in range(100):
    sample = np.random.choice(all_people, size=1000)
    zms = sm.stats.DescrStatsW(sample).zconfint_mean(alpha=0.05)
    print(zms)

#    mean=np.mean(sample)
#    values.append(mean)

# print(np.mean(values))
