# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from pylab import rcParams
import matplotlib.pyplot as plt
import warnings
import itertools
import statsmodels.api as sm
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
# for dirname, _, filenames in os.walk('./kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


train_param = {
    "lookBack": 30,
    "units": 50,
    "epochs": 70,
    "batchSize": 15,
    "reduce_lr": 5,
    "predict_size": 90,
    "drop_out": 0.1,
}


# Convert date coulmns to specific format
def dateparse(x):
    # print('x:', x)
    # print('pd:', pd.datetime.strptime(x, '%d-%b-%y'))
    # x = 15-Dec-93
    # pd.datetime.strptime(x, '%d-%b-%y') = 1993-12-15 00:00:00
    # return pd.datetime.strptime(x, '%d-%b-%y')
    return pd.datetime.strptime(x, '%Y/%m/%d')

# Read dataframe info


def DfInfo(df_initial):
    # gives some infos on columns types and numer of null values
    tab_info = pd.DataFrame(df_initial.dtypes).T.rename(
        index={0: 'column type'})
    tab_info = tab_info.append(pd.DataFrame(
        df_initial.isnull().sum()).T.rename(index={0: 'null values (nb)'}))
    tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum() / df_initial.shape[0] * 100).T.
                               rename(index={0: 'null values (%)'}))
    return tab_info


# convert an array of values into a data_set matrix def
def create_data_set(_data_set, _look_back=1):
    data_x, data_y = [], []
    # print(_data_set)
    # shape回傳的為 dimension, elements
    # 3611維，1個維度1個element
    # print(_data_set.shape)
    # print('\n')
    # print(len(_data_set))
    # print(range(len(_data_set) - _look_back - 1))
    # 計算每次抓取look_back個次數，可以做幾次loop (減1應該是怕剛好相同數量)
    # 代表過去 _look_back 天的資訊
    for i in range(len(_data_set) - _look_back - 1):
        index = i + _look_back
        data_x.append(_data_set[i:index, 0])
        data_y.append(_data_set[index, 0])
    return np.array(data_x), np.array(data_y)


# Read csv file
df = pd.read_csv(r'./oilData/CPC.csv',
                 parse_dates=['Date'], date_parser=dateparse)
df = df.dropna()
# print(df.head(10))

# Sort dataset by column Date
df = df.sort_values('Date')
# print(df)

# # 相同Date的Price會sum起來
df = df.groupby('Date')['92OilPrice'].sum().reset_index()

df.set_index('Date', inplace=True)

# # # print(datetime.date(year=2000, month=1, day=1))
# # # 只留下index為日期2000-01-01之後的日期
# # df = df.loc[datetime.date(year=2000, month=1, day=1):]

# # # df = df.loc[datetime.date(year=2000, month=1, day=1):datetime.date(year=2002, month=1, day=1)]
# # # print(df)

# MS意思是MonthBegin，M為MonthEnd
# ref: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
# 此行意思為，重新採樣每個月份的平均值，並轉為月份-月份初始日
# y = df['92OilPrice'].resample('M').mean()
# print(y)

# figsize: a tuple (width, height) in inches
# y.plot(figsize=(15, 6))
# # plt.savefig("1-PriceMeanGroupByMonthStart.png")

# # 顯示圖像的最大範圍 width, height in inches
# rcParams['figure.figsize'] = [18, 8]
# # # 數據的整體趨勢 使用季節性分解
# decomposition = sm.tsa.seasonal_decompose(y, model='additive')
# fig = decomposition.plot()
# plt.savefig("2-SeasonalDecompose.png")

# # normalize the data_set
# # 最小最大值標準化 0~1之間
sc = MinMaxScaler(feature_range=(0, 1))
# print(df)
df = sc.fit_transform(df)
# # print(len(df))

# split into train and test sets
# 訓練集為目前df長度之70%，測試集為目前df長度之30%
# train_size = int(len(df) * 0.80)
# test_size = len(df) - train_size
# train, test = df[0:train_size, :], df[train_size:len(df), :]
print('complete length: ', len(df))
train_size = int(len(df)) - train_param["predict_size"]  # int(len(df) * 0.80)
print('train_size: ', 0, ' to ', train_size)
test_size = train_param["predict_size"]  # len(df) - train_size
print('test_size: ', len(df)-test_size, 'to', len(df))
print('test_size length: ', test_size)
# # train拿前70%的資料量，test拿後30%資料量
train = df[0:train_size, :]
test = df[len(df)-test_size: len(df), :]
# # print('\n')
# # # print(train)

# # reshape into X=t and Y=t+1
look_back = train_param["lookBack"]
X_train, Y_train, X_test, Y_test = [], [], [], []
X_train, Y_train = create_data_set(train, look_back)
# # # print(X_train.shape, Y_train.shape)
# # # # print(X_train)
# # # 因為現在 X_train 是 2-dimension，將它 reshape 成 3-dimension: [price, look_back, indicators]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# # # print(X_train)
X_test, Y_test = create_data_set(test, look_back)
# # # # 把X_test的資料內容重新shape為3維陣列
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


if not os.path.isfile('./finaltest.h5'):
    print('\n======no model, start to train model======\n')
    # # create and fit the LSTM network regressor = Sequential()
    # 宣告一個RNN的Model網路層
    regressor = Sequential()

    # units: 神經元的數目
    # 第一層的 LSTM Layer 記得要設定input_shape參數
    units = train_param["units"]
    regressor.add(LSTM(units=units, return_sequences=True,
                       input_shape=(X_train.shape[1], 1)))
    # Dropout是一种训练时可以采用的正则化方法，通过在正向传递和权值更新的过程中对LSTM神经元的输入和递归连接进行概率性失活，该方法能有效避免过拟合并改善模型性能，避免overfitting
    regressor.add(Dropout(train_param["drop_out"]))

    # 加入第二層 LSTM Layer & Dropout
    # regressor.add(LSTM(units=units, return_sequences=True))
    # regressor.add(Dropout(train_param["drop_out"]))

    # 加入第三層 LSTM Layer & Dropout  LSTM Layer 即將跟 Ouput Layer 做連接，因此注意這邊的 return_sequences 設為預設值 False （也就是不用寫上 return_sequences）
    regressor.add(LSTM(units=units))
    regressor.add(Dropout(train_param["drop_out"]))

    # 新增Ouput Layer: units 設為 1
    regressor.add(Dense(units=1))

    # Compiling & Fitting LSTM model
    # Adam為目前較常使用的Optimizer
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    # ReduceLROnPlateau是為了讓訓練過程中不斷縮小學習率，可以快速又精確的獲得最優模型
    # monitor：監測的值，可以是accuracy，val_loss,val_accuracy
    # patience：當patience個epoch過去而模型性能不提升時，學習率减少的動作會被觸發
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', patience=train_param["reduce_lr"])
    # 進行訓練
    # batch_size :整數或None。每次梯度更新的樣本數。如果未指定，默認為32。
    # epochs :整數。訓練模型迭代輪次。一個輪次是在整個x和y上的一輪迭代。請注意，與initial_epoch一起，epochs被理解為「最終輪次」。模型並不是訓練了epochs輪，而是到第epochs輪停止訓練。
    # validation_data: 用來評估損失，以及在每輪結束時的任何模型度量指標
    # shuffle: 是否在每輪迭代之前混洗數據，這邊填false，資料必須保有順序不能打散
    history = regressor.fit(X_train, Y_train, epochs=train_param["epochs"], batch_size=train_param["batchSize"], validation_data=(
        X_test, Y_test), callbacks=[reduce_lr], shuffle=False)

    plt.figure(figsize=(8, 4))
    # history.history['loss']為連續epoch訓練集損失值
    plt.plot(history.history['loss'], label='Train Loss')
    print('Train Loss', history.history['loss'])
    # history.history['loss']為連續epoch驗證集損失值
    plt.plot(history.history['val_loss'], label='Test Loss')
    print('\n')
    print('Test Loss', history.history['val_loss'])

    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    # legend設定位置於右上
    plt.legend(loc='upper right')
    plt.savefig("3-HistoryLoss.png")
else:
    print('\n======has model, just load Model.======\n')
    regressor = load_model('./finaltest.h5')


# 使用訓練集來預測
train_predict = regressor.predict(X_train)
# 使用測試集來預測
test_predict = regressor.predict(X_test)
regressor.save('./finaltest.h5')

# # invert predictions
# 將降維後的數據轉換成原始數據
train_predict = sc.inverse_transform(train_predict)
Y_train = sc.inverse_transform([Y_train])
test_predict = sc.inverse_transform(test_predict)
Y_test = sc.inverse_transform([Y_test])

print('Train Mean Absolute Error:', mean_absolute_error(
    Y_train[0], train_predict[:, 0]))
print('Train Root Mean Squared Error:', np.sqrt(
    mean_squared_error(Y_train[0], train_predict[:, 0])))
print('Test Mean Absolute Error:', mean_absolute_error(
    Y_test[0], test_predict[:, 0]))
print('Test Root Mean Squared Error:', np.sqrt(
    mean_squared_error(Y_test[0], test_predict[:, 0])))

# # Compare Actual vs. Prediction
# 設定x軸資料(180筆資料)，為0~179的1維array
# print(Y_test.shape[1])
length = Y_test.shape[1]
print('xAxis length:', length)
xAxisData = [x for x in range(length)]
plt.figure(figsize=(8, 4))
plt.plot(xAxisData, Y_test[0][:length], marker='.', label="actual")
plt.plot(xAxisData, test_predict[:, 0][:length], 'r', label="prediction")

# 調整子圖之間及其周圍的填充
plt.tight_layout()
# 從plot()函數中移除頂部的邊框
sns.despine(top=True)
# 子圖距離figure左邊的距離
# plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.savefig("4-PredictionResult.png")
