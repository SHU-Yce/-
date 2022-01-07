import cydtw
import numpy as np
import xlrd
import math
from sklearn.preprocessing import MinMaxScaler

filepath = './data_filled.xls'

work_book = xlrd.open_workbook(filepath)
table = work_book.sheet_by_index(0)
print(">>>数据集长度：", table.nrows-1)
full_data_0, full_data_1 = [], []
for i in range(1, table.nrows):
    rowValue = table.row_values(i)
    if rowValue[-1] == 0:
        full_data_0.append(rowValue)
    else:
        full_data_1.append(rowValue)

full_data_0 = np.array(full_data_0, dtype=np.float64).reshape(106, 8)
np.random.shuffle(full_data_0)
full_data_1 = np.array(full_data_1, dtype=np.float64).reshape(121, 8)
np.random.shuffle(full_data_1)

train = np.r_[full_data_0[:73, :], full_data_1[:84, :]]           # 取70%的数据作为训练集，30%为测试集
test = np.r_[full_data_0[73:, :], full_data_1[84:, :]]
train_labels = train[:, 7:]
test_labels = test[:, 7:]
min_max_scaler = MinMaxScaler()
train_data = min_max_scaler.fit_transform(train[:,:7])
test_data = min_max_scaler.fit_transform(test[:,:7])

# train_data = train[:, :7]
# test_data = test[:, :7]


def gaussian(dist, a=1, b=0., c=0.15):                       # 高斯权重函数
    return a * math.e ** (-(dist - b) ** 2 / (2 * c ** 2))

def predict(K,train_data,train_labels,test_data,test_labels):
    i=0
    accuracy=0
    predict_labels = []
    for test in test_data:
        t_dis=[]
        test = test.reshape(7, 1)
        for train in train_data:
           train = train.reshape(7, 1)
           dis=cydtw.dtw(test, train)#dtw计算距离
           t_dis.append(dis) #距离数组
        #KNN算法预测标签
        w_a, w_0, w_1 = [], [], []
        for id in np.argpartition(t_dis, K)[:K]:
            w = gaussian(t_dis[id])
            w_0.append(w) if train_labels[id] == 0 else w_1.append(w)
            w_a.append(w)
        # print(w_a)
        f_0 = np.sum(w_0) / np.sum(w_a)
        f_1 = np.sum(w_1) / np.sum(w_a)
        # nearest_series_labels = np.array(train_labels[np.argpartition(t_dis, K)[:K]]).astype(int).reshape(K,)
        preditc_labels_single = 0 if f_0 > f_1 else 1
        predict_labels.append(preditc_labels_single)
        #计算正确率
        if preditc_labels_single==test_labels[i] :
            accuracy+=1
        # else:
        #     print(test)
        i+=1
    print('The accuracy is %f (%d of %d)'%((accuracy/test_data.shape[0]),accuracy,test_data.shape[0]))
    return accuracy/test_data.shape[0]

predict(5,train_data,train_labels,test_data,test_labels)


