#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang_wen_hui
@file: priceForecast.py
@time: 2020/3/21 17:53
@software: PyCharm
"""
# 导入warnings包，利用过滤器来实现忽略警告语句
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import scipy.stats as st



class UserCarForecast():

    def __init__(self):
        self.Train_data = pd.read_csv('../data/used_car_train_20200313.csv', sep=',')
        self.Test_data = pd.read_csv("../data/used_car_testA_20200313.csv", sep=',')

    """
    2.载入数据：
          ·载入训练集和测试集
          ·简略观察数据(head()+shape)
    """
    # 载入训练集和测试集：
    def loadData(self):
        print("Train data shape: ",self.Train_data.shape)
        print("TestA data shape: ",self.Test_data.shape)
        print(self.Train_data.head(10))
        ## 简略观察数据(head()+shape)
        print(self.Train_data.head().append(self.Train_data.tail()))
        print(self.Train_data.shape)
        print(self.Test_data.head().append(self.Test_data.tail()))
        print(self.Test_data.shape)
        return None


    """
    3. 数据总览：
           ·通过describe()来熟悉数据的相关统计量
           ·通过info()来熟悉数据类型
    """
    def watchData(self):
        print(self.Train_data.describe())
        print(self.Test_data.describe())
        print(self.Train_data.info())
        print(self.Test_data.info())
        return None


    """
    4. 判断数据缺失和异常
        ·查看每列的存在nan情况
        ·异常值检测
    """
    def judgeData(self):
        # 查看每列存在的nan情况
        # print(self.Train_data.isnull().sum())
        # print(self.Test_data.isnull().sum())
        # nan可视化
        missing=self.Train_data.isnull().sum()
        missing=missing[missing>0]
        missing.sort_values(inplace=True)
        missing.plot.bar()
        # 可视化看下缺省值
        # msno.matrix(self.Train_data.sample(250))
        # msno.bar(self.Train_data.sample(1000))
        # plt.show()
        # 查看异常值检测
        # print(self.Train_data.info())
        # print(self.Train_data['gearbox'].value_counts())
        # print(self.Train_data['power'].value_counts())
        # print(self.Train_data['kilometer'].value_counts())
        # print(self.Train_data['notRepairedDamage'].value_counts())
        self.Train_data['gearbox'].replace('-',np.nan,inplace=True)
        # print(self.Train_data['gearbox'].value_counts())
        self.Train_data['power'].replace('-',np.nan,inplace=True)
        # print(self.Train_data['power'].value_counts())
        self.Train_data['kilometer'].replace('-',np.nan,inplace=True)
        # print(self.Train_data['kilometer'].value_counts())
        self.Train_data['notRepairedDamage'].replace('-',np.nan,inplace=True)
        # print(self.Train_data['notRepairedDamage'].value_counts())
        # print(self.Train_data["seller"].value_counts())
        # print(self.Train_data["offerType"].value_counts())
        del self.Train_data["seller"]
        del self.Train_data["offerType"]
        del self.Test_data["seller"]
        del self.Test_data["offerType"]
        return self.Train_data,self.Test_data


    """
    5. 了解预测值的分布
       ·总体分布概况(无界约翰逊分布等)
       ·查看skewness and kurtosis
       ·查看预测值的具体频数
    """
    def knowledgePrediction(self):
        np.set_printoptions(suppress=True)
        Train_data,Test_data=self.judgeData()
        # 有负值，可以去掉
        print(Train_data['price'].value_counts())
        y=Train_data['price']
        plt.figure(1)
        return None

"""
6. 特征分为类别特征和数字特征，并对类别特征查看unique分布
"""

"""
7. 数字特征分析
  ·相关性分析
  ·查看几个特征的偏度和峰值
  ·每个特征的分布可视化
  ·数字特征相互之间的关系可视化
  ·多变量互相回归关系可视化
"""

"""
8. 类别特征分析
   ·unique分布
   ·类别特征箱形图可视化
   ·类别特征的柱状图可视化类别
   ·特征的每个类别频数可视化(count_plot)
"""

"""
9.用pandas_profiling生成数据报告
"""

if __name__ == '__main__':
    userCarForecast=UserCarForecast()
    userCarForecast.knowledgePrediction()
