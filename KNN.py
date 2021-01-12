import os
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import joblib
import math
import scipy.io as scio
from ecgdetectors import Detectors
from Denoise import Denoise
import pandas as pd
from copy import deepcopy

Category_mapping={
    '窦性心律_左室肥厚伴劳损':0,
    '窦性心动过缓':1,'窦性心动过速':2,
    '窦性心律_不完全性右束支传导阻滞':3,
    '窦性心律_电轴左偏':4,
    '窦性心律_提示左室肥厚':5,
    '窦性心律_完全性右束支传导阻滞':6,
    '窦性心律_完全性左束支传导阻滞':7,
    '窦性心律_左前分支阻滞':8,
    '正常心电图':9
}
detectors=Detectors(500)

def split_train_val(train_path, r = 0.7):
    cls_list = os.listdir(train_path)
    data_train = []
    data_val = []
    label_train = []
    label_val = []

    for cls_name in cls_list:
        cls_id = Category_mapping[cls_name]
        cls_path = os.path.join(train_path, cls_name)
        record_list = os.listdir(cls_path)
        hb_list = []
        num_hb = 0
        for record_file in record_list:
            data = scio.loadmat(os.path.join(cls_path, record_file))
            hb = data['Beats'][0, 0]['beatData'][0]
            num_hb+=hb.size
            for single_hb in hb:
                hb_list.append(single_hb)
        num_train = round(num_hb*r)
        data_train+=hb_list[:num_train]
        data_val += hb_list[num_train:]
        label_train += [cls_id]*num_train
        label_val += [cls_id]*(num_hb-num_train)
    return data_train, label_train, data_val, label_val

def hb_split(hb):
    r_peaks = detectors.swt_detector(hb[:, 0])
    num_hb = len(r_peaks) # R峰数量（心跳数量）
    all_hb = []
    s = 0 # 切分起始点
    e = 0 # 切分终点
    for i in range(num_hb-1):
        e = round(2/3*(r_peaks[i+1]-r_peaks[i]))+r_peaks[i]
        all_hb.append(hb[s:e, :])
        s = e
    all_hb.append(hb[s:, :])
    return all_hb

def collect_test(test_path):
    record_list = os.listdir(test_path)
    name_list = [] # 文件名列表，无后缀
    data_test = []
    for record_file in record_list:
        record_name = os.path.splitext(record_file)[0]
        name_list.append(record_name)
        hb = scio.loadmat(os.path.join(test_path, record_file))['data'][:, 1:]
        splited = hb_split(hb) # list 包含该条记录所有心跳
        data_test.append(splited)
    return data_test, name_list

def denoise(data):
    clean_data = deepcopy(data)
    for i in range(len(clean_data)):
        Denoise(clean_data[i])
    return clean_data

def features_time(data,p1,p2):
    #均值
    df_mean=data[p1:p2].mean()
    #方差
    df_var=data[p1:p2].var()
    #标准差
    df_std=data[p1:p2].std()
    #均方根
    df_rms=np.sqrt(pow(df_mean,2) + pow(df_std,2))# 改用np.sqrt
    #偏度
    df_skew=data[p1:p2].skew()
    #峭度
    df_kurt=data[p1:p2].kurt()
    sum= np.sum(np.sqrt(abs(data)))
    #波形因子
    df_boxing=df_rms / (abs(data[p1:p2]).mean())
    #峰值因子
    df_fengzhi=(np.max(data[p1:p2])) / df_rms
    #脉冲因子
    df_maichong=(np.max(data[p1:p2])) / (abs(data[p1:p2]).mean())
    #裕度因子
    df_yudu=(np.max(data[p1:p2])) / pow((sum/(p2-p1)),2)
    featuretime_list = [df_mean,df_rms,df_skew,df_kurt,df_boxing,df_fengzhi,df_maichong,df_yudu]
    features = np.array([x.values for x in featuretime_list]).T.reshape(-1)
    return features

def gen_feature(data):
    return np.array([features_time(pd.DataFrame(hb), 0, hb.shape[0]) for hb in data])

def eval(model, data_val, label_val):
    pred_val = model.predict(data_val)
    acc = np.sum(pred_val==label_val)/len(label_val)
    print("Val Acc:", acc)

def test(model, data_test, name_list, mean, std):
    result = {'id':[], 'categories':[]}
    for i in range(len(name_list)):
        result['id'].append(name_list[i])
        data = data_test[i]
        # 降噪先
        clean = denoise(data)
        feature = gen_feature(clean)
        feature_norm = (feature-mean)/std
        prob = model.predict_proba(feature_norm)
        predict = np.argmax(np.sum(prob, axis=0))
        result['categories'].append(predict)
    return result

def writeincsv(result, file_name):
    df = pd.DataFrame(result)
    df.to_csv(file_name, index=False)

def main(train_path, test_path, dir_csv, n_neighbors):
    print("KNN for ECG Classification!")
    # Load data
    print("Loading the training data...")
    data_train, label_train, data_val, label_val = split_train_val(train_path)
    clean_train = denoise(data_train)
    clean_val = denoise(data_val)
    feature_train = gen_feature(clean_train)
    feature_val = gen_feature(clean_val)

    # Normalization
    MEAN = np.mean(feature_train, axis=0)
    STD = np.std(feature_train, axis=0)
    feature_train_norm = (feature_train-MEAN)/STD
    feature_val_norm = (feature_val-MEAN)/STD

    # Construct KNN classifier, train and eval
    K = n_neighbors
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(feature_train_norm, label_train)
    eval(neigh, feature_val_norm, label_val)

    # Test
    print("Loading the test data...")
    data_test, name_list = collect_test(test_path)
    print("Loading completed!")
    print("Testing...")
    result = test(neigh, data_test, name_list, MEAN, STD)

    writeincsv(result, dir_csv)
    print("Completed!")
if __name__ == "__main__":
    main(train_path, test_path, dir_csv, n_neighbors)