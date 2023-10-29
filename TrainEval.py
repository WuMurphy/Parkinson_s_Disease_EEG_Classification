import csv
import operator
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.svm as svm
import scipy.io as scio

import mne
import pywt

from scipy.io import loadmat
from statistical_feature_generate_wpd import segmentSignals, computeFeature
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFpr
from sklearn.metrics import classification_report, roc_auc_score, plot_roc_curve,\
    average_precision_score, plot_precision_recall_curve
from mne.preprocessing import ICA


def featureSelection(data, label):
    selector = SelectFpr(alpha=0.05)
    new_data = selector.fit_transform(data, label)
    return new_data


def correlatioCalculator(data):
    df = pd.DataFrame(data)
    print(df.corr())


def loadData(path):
    dataFeature = []
    dataLabel = []
    filelist = os.listdir(path)
    filelist.sort()
    for file in filelist:
        person = np.load(path + file, allow_pickle=True).item()
        dataLabel.append(1 if person['label']==0 else 0)
        features = []
        data = person['source']
        dataShape = data.shape
        for i in range(dataShape[0]):
            for j in range(dataShape[1]):
                for k in range(dataShape[2]):
                    for key in data[i, j, k].keys():
                        if not (key == 'bandLabel' or key == 'channelName'):
                            features.append(data[i, j, k][key])
        dataFeature.append(features)
    dataFeature = np.array(dataFeature)

    dataFeature = preprocessing.normalize(dataFeature, axis=0)

    return dataFeature, dataLabel


def loadData_singelChannel(path, channelIndex):
    dataFeature = []
    dataLabel = []
    filelist = os.listdir(path)
    filelist.sort()
    for file in filelist:
        person = np.load(path + file, allow_pickle=True).item()
        dataLabel.append(1 if person['label']==0 else 0)
        features = []
        data = person['source']
        dataShape = data.shape
        for j in range(dataShape[1]):
            for k in range(dataShape[2]):
                # band_features = []
                for key in data[channelIndex, j, k].keys():
                    if not (key == 'channelName' or key == 'bandLabel' or key == 'coefficientLabel'):
                        features.append(data[channelIndex, j, k][key])
                # band_features = np.array(band_features)
                # band_features = preprocessing.scale(band_features, axis=0)
                # features.append(item for item in band_features)
        dataFeature.append(features)
    dataFeature = np.array(dataFeature)

    dataFeature = preprocessing.scale(dataFeature, axis=0)

    return dataFeature, dataLabel, data[channelIndex, 0, 0]['channelName']


def loadData_bestChannel(path, bestChannel):
    dataFeature = []
    dataLabel = []
    filelist = os.listdir(path)
    filelist.sort()
    for file in filelist:
        person = np.load(path + file, allow_pickle=True).item()
        dataLabel.append(1 if person['label']==0 else 0)
        features = []
        data = person['source']
        dataShape = data.shape
        for i in range(dataShape[0]):
            if data[i, 0, 0]['channelName'] in bestChannel:
                for j in range(dataShape[1]):
                    for k in range(dataShape[2]):
                        for key in data[i, j, k].keys():
                            if not (key == 'bandLabel' or key == 'channelName' or key == 'coefficientLabel'):
                                features.append(data[i, j, k][key])
        dataFeature.append(features)
    dataFeature = np.array(dataFeature)

    dataFeature = preprocessing.scale(dataFeature, axis=0)

    return dataFeature, dataLabel


def svmEval(trainFeature, trainLabel, evalFeature, evalLabel):
    clf = svm.SVC(C=10, kernel='linear', probability=True)
    clf.fit(trainFeature, trainLabel)
    pred = clf.predict(evalFeature)

    result = classification_report(evalLabel, pred)
    prob = clf.predict_proba(evalFeature)[:, 1]

    auc_score = roc_auc_score(y_true=evalLabel, y_score=prob)

    ap_score = average_precision_score(y_true=evalLabel, y_score=prob)

    plot_precision_recall_curve(estimator=clf, X=evalFeature, y=evalLabel, pos_label=1)

    plot_roc_curve(estimator=clf, X=evalFeature, y=evalLabel, pos_label=1)
    plt.show()

    print("AUC Score: ", auc_score)
    print("AP Score: ", ap_score)

def randomForest(trainFeature, trainLabel, evalFeature, evalLabel):
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(trainFeature, trainLabel)
    score = rfc.score(evalFeature, evalLabel)
    print(score)


def rfEval(trainFeature, trainLabel, evalFeature, evalLabel):
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(trainFeature, trainLabel)
    pred = rfc.predict(evalFeature)

    result = classification_report(evalLabel, pred)
    prob = rfc.predict_proba(evalFeature)[:, 1]


    auc_score = roc_auc_score(y_true=evalLabel, y_score=prob)

    ap_score = average_precision_score(y_true=evalLabel, y_score=prob)

    plot_precision_recall_curve(estimator=rfc, X=evalFeature, y=evalLabel, pos_label=1)

    plot_roc_curve(estimator=rfc, X=evalFeature, y=evalLabel, pos_label=1)
    plt.show()

    print("AUC Score: ", auc_score)
    print("AP Score: ", ap_score)


def singleChannel_eval(train_rootPath, eval_rootPath):
    singleChannelResult = {}

    for channelIndex in range(63):
        trainFeature_single, trainLabel_single, channelName = loadData_singelChannel(train_rootPath, channelIndex)
        evalFeature_single, evalLabel_single, channelName = loadData_singelChannel(eval_rootPath, channelIndex)

        clf = svm.SVC(C=10, kernel='linear', probability=True)
        clf.fit(trainFeature_single, trainLabel_single)
        singleChannelResult[channelName] = clf.score(evalFeature_single, evalLabel_single)

    with open('Single channel performance(EyesClose wptBands2000clipLength).csv', "w") as f:
        writer = csv.writer(f)
        for key, value in singleChannelResult.items():
            writer.writerow([key, value])
    # Plot
    # draw_from_dict(singleChannelResult, heng=1, reverse=True)


def draw_from_dict(dicData, RANGE=-1, heng=0, reverse=False):
    """
    :param dicData: 字典的数据
    :param RANGE: 截取显示的字典的长度
    :param heng: 代表条状图的柱子是竖直向上的。heng=1，代表柱子是横向的。考虑到文字是从左到右的，让柱子横向排列更容易观察坐标轴
    :return:
    """
    by_value = sorted(dicData.items(), key=lambda item: item[1], reverse=reverse)
    x = []
    y = []
    for d in by_value:
        x.append(d[0])
        y.append(d[1])
    if heng == 0:
        plt.bar(x[0:RANGE], y[0:RANGE])
        plt.show()
        return
    elif heng == 1:
        plt.barh(x[0:RANGE], y[0:RANGE])
        plt.show()
        return
    else:
        return "heng的值仅为0或1！"



if __name__ == '__main__':

    train_rootPath = "/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesClose_AllChannels_AllFeatures_2000clipLength_15segments_wptBands/train/"
    eval_rootPath = "/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesClose_AllChannels_AllFeatures_2000clipLength_15segments_wptBands/eval/"

    # bestChannel = ['Oz', 'F4', 'P8', 'CP2', 'Cz', 'Fp2', 'P2', 'FC5', 'T7', 'O1', 'FC6', 'PO4']
    bestChannel_wpt = ['TP7', 'TP9', 'TP10', 'FC3', 'FC4',
                       'FC5', 'PO3', 'PO4', 'PO7', 'PO8',
                       'CP3', 'CP5', 'Pz', 'P1', 'P3',
                       'P4', 'P5', 'P6', 'P8', 'AF3',
                       'C4', 'F5', 'Oz', 'O1', 'O2',
                       'F4', 'CP2', 'Cz', 'Fp2', 'P2', 'T7', 'FC6']

    # trainFeature, trainLabel = loadData_bestChannel(train_rootPath, bestChannel_wpt)
    # evalFeature, evalLabel = loadData_bestChannel(eval_rootPath, bestChannel_wpt)

    # Single Channel evaluation
    singleChannel_eval(train_rootPath, eval_rootPath)

    # correlatioCalculator(trainFeature)

    # svmEval(trainFeature, trainLabel, evalFeature, evalLabel)
