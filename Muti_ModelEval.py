# 导入库
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, \
    roc_curve
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc

from TrainEval import loadData_bestChannel

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# 数据集分割
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8675309)


# 绘制roc曲线
def calculate_auc(y_test, pred):
    print("auc:", roc_auc_score(y_test, pred))
    fpr, tpr, thersholds = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k-', label='ROC (area = {0:.2f})'.format(roc_auc), color='blue', lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.show()


# 使用Yooden法寻找最佳阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


# 计算roc值
def ROC(label, y_prob):
    fpr, tpr, thresholds = roc_curve(label, y_prob, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    optimal_threshold, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_threshold, optimal_point


# 计算混淆矩阵
def calculate_metric(label, y_prob, optimal_threshold):
    p = []
    for i in y_prob:
        if i >= optimal_threshold:
            p.append(1)
        else:
            p.append(0)
    confusion = confusion_matrix(label, p)
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Accuracy = (TP + TN) / float(TP + TN + FP + FN)
    Sensitivity = TP / float(TP + FN)
    Specificity = TN / float(TN + FP)
    return Accuracy, Sensitivity, Specificity


# 10-fold cross validation
def tenFold(X, y, name, model, save_path, fold=3):
    results = []
    for this_i in range(10):
        kf = KFold(n_splits=fold, shuffle=True)
        # plt.figure(figsize=(10, 10))
        lw = 2
        for (train, test) in kf.split(X=X):
            X_train = X[train, :]
            y_train = [y[index] for index in train]
            X_test = X[test, :]
            y_test = [y[index] for index in test]

            clf = model.fit(X_train, y_train)
            # results.append(clf.score(X_test, y_test))
            Score = clf.score(X_test, y_test)
            pred_proba = clf.predict_proba(X_test)
            y_prob = pred_proba[:, 1]
            fpr, tpr, roc_auc, Optimal_threshold, optimal_point = ROC(y_test, y_prob)
            Accuracy, Sensitivity, Specificity = calculate_metric(y_test, y_prob, Optimal_threshold)
            result = [Optimal_threshold, Accuracy, Sensitivity, Specificity, roc_auc, Score, this_i + 1]
            results.append(result)
        # plt.plot(fpr, tpr, lw=lw)
        # print(name + ": " + str(Accuracy))

    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic Curve (XGBoost ' + str(fold) + '-fold cross validation)')
    # plt.savefig(save_path, dpi=300)
    # mean_score = sum(results) / len(results)
    df = pd.DataFrame(results)
    df.columns = ["Optimal_threshold", "Accuracy", "Sensitivity", "Specificity", "AUC_ROC", "R2 Score",
                  "Round Number"]
    df.to_csv("29channels_10-round-3-fold_results_UNM.csv")
    # print(mean_score)


def diffTestSet(X, y, name, model):
    local_results = []
    local_roc_ = []

    kf = KFold(n_splits=3, shuffle=True)
    for (train, test) in kf.split(X=X):
        X_train = X[train, :]
        y_train = [y[index] for index in train]
        X_test = X[test, :]
        y_test = [y[index] for index in test]

        clf = model.fit(X_train, y_train)
        Score = clf.score(X_test, y_test)
        pred_proba = clf.predict_proba(X_test)
        y_prob = pred_proba[:, 1]
        fpr, tpr, roc_auc, Optimal_threshold, optimal_point = ROC(y_test, y_prob)
        Accuracy, Sensitivity, Specificity = calculate_metric(y_test, y_prob, Optimal_threshold)
        result = [Optimal_threshold, Accuracy, Sensitivity, Specificity, roc_auc, Score, name]
        local_results.append(result)
        local_roc_.append([fpr, tpr, roc_auc, name])
        print(name + ": " + str(Accuracy))
        print("Optimal threshold: " + str(Optimal_threshold) +
              ", Optimal False Positive rate: " + str(optimal_point[0]) +
              ", Optimal True Positive rate: " + str(optimal_point[1]))

    df_result = pd.DataFrame(local_results)
    df_result.columns = ["Optimal_threshold", "Accuracy", "Sensitivity", "Specificity", "AUC_ROC", "R2 Score",
                         "Model_name"]
    df_result.to_csv("BestModelsCompare_EyesClose_32bestChannels_47520bestFeaturesFClassIF_wpt&IIR.csv")


def loadData(path):
    dataFeature = []
    dataLabel = []
    filelist = os.listdir(path)
    filelist.sort()
    for file in filelist:
        person = np.load(path + file, allow_pickle=True).item()
        dataLabel.append(1 if person['label'] == 0 else 0)
        features = []
        data = person['source']
        dataShape = data.shape
        for i in range(dataShape[0]):
            for j in range(dataShape[1]):
                for k in range(dataShape[2]):
                    for key in data[i, j, k].keys():
                        if not (key == 'bandLabel' or key == 'channelName' or key == 'coefficientLabel'):
                            features.append(data[i, j, k][key])
        dataFeature.append(features)
    dataFeature = np.array(dataFeature)

    dataFeature = preprocessing.scale(dataFeature, axis=0)

    return dataFeature, dataLabel


def combineWptAndIIR_Features(wpt_train_path, wpt_eval_path, iir_train_path, iir_eval_path, bestChannel=None):
    if bestChannel == None:
        wpt_X_train, wpt_y_train = loadData(wpt_train_path)
        wpt_X_eval, wpt_y_eval = loadData(wpt_eval_path)
        iir_X_train, iir_y_train = loadData(iir_train_path)
        iir_X_eval, iir_y_eval = loadData(iir_eval_path)
    else:
        wpt_X_train, wpt_y_train = loadData_bestChannel(wpt_train_path, bestChannel)
        wpt_X_eval, wpt_y_eval = loadData_bestChannel(wpt_eval_path, bestChannel)
        iir_X_train, iir_y_train = loadData_bestChannel(iir_train_path, bestChannel)
        iir_X_eval, iir_y_eval = loadData_bestChannel(iir_eval_path, bestChannel)

    this_X_train = np.concatenate((wpt_X_train, iir_X_train), axis=1)
    this_y_train = wpt_y_train
    this_X_test = np.concatenate((wpt_X_eval, iir_X_eval), axis=1)
    this_y_test = wpt_y_eval

    return this_X_train, this_y_train, this_X_test, this_y_test


def adaboostParameterOptimizer(X_train, y_train):
    # param_test1 = {"n_estimators": range(50, 300, 50)}
    #
    estimatorCart = DecisionTreeClassifier(max_depth=1)
    # gsearch1 = GridSearchCV(estimator=AdaBoostClassifier(estimatorCart), param_grid=param_test1, scoring="roc_auc", cv=5)
    #
    # gsearch1.fit(X_train, y_train)
    #
    # print(gsearch1.best_params_, gsearch1.best_score_)

    n_estimator1 = 180
    param_test2 = {"n_estimators": range(n_estimator1 - 10, n_estimator1 + 10, 1)}
    gsearch2 = GridSearchCV(estimator=AdaBoostClassifier(estimatorCart),
                            param_grid=param_test2, scoring="roc_auc", cv=5)
    gsearch2.fit(X_train, y_train)
    print(gsearch2.best_params_, gsearch2.best_score_)


def xgBoostParameterOptimizer(X_train, y_train):
    xgb1 = XGBClassifier(
        booster='gblinear',
        scilent=True,
        learning_rate=0.01,
        gamma=0.03,
        n_estimators=140,
        max_depth=13,
        min_samples_split=2,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27,
        subsample=0.6,
        colsample_bytree=0.8,
        reg_alpha=0
    )

    # param_test1 = {
    #     'max_depth': range(11, 15, 1),                              # Result: 3 for gbtree, 14 for gblinear
    #     'min_child_weight': [1, 2]                                  # Result: 3 for gbtree, 2 for gblinear
    # }
    # param_test2 = {'gamma': [i/100.0 for i in range(0, 100)]}       # Result: 0.58 for gbtree, 0.03 for gblinear
    # param_test3 = {
    #     'subsample': [i / 10.0 for i in range(1, 10)],              # Result: 0.6 for gbtree, 0.6 for gblinear
    #     'colsample_bytree': [j / 10.0 for j in range(1, 10)]        # Result: 0.8 for gbtree, 0.8 for gblinear
    # }
    # param_test4 = {
    #     'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]                  # Result: 0.01 for gbtree, 0 for gblinear
    # }
    param_test5 = {
        'learning_rate': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]  # Result: 0.005 for gbtree, 0.01 for gblinear
    }
    gsearch1 = GridSearchCV(estimator=xgb1, param_grid=param_test5, scoring='roc_auc', n_jobs=4, cv=18)
    gsearch1.fit(X_train, y_train)
    print(gsearch1.best_params_, gsearch1.best_score_)


if __name__ == '__main__':

    train_rootPath_close = "/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesClose_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands/train/"
    eval_rootPath_close = "/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesClose_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands/eval/"
    train_rootPath_open = "/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesOpen_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands/train/"
    eval_rootPath_open = "/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesOpen_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands/eval/"

    # bestChannel = ['TP7', 'FC5', 'PO4', 'CP5', 'P3', 'AF3', 'P4', 'P8']

    bestChannel_5 = ['Oz', 'P8', 'FC5', 'O1', 'PO4']

    bestChannel_12 = ['Oz', 'F4', 'P8', 'CP2', 'Cz', 'Fp2', 'P2', 'FC5', 'T7', 'O1', 'FC6', 'PO4']

    bestChannel_25 = ['TP7', 'TP9', 'TP10', 'FC3', 'FC4',
                      'FC5', 'PO3', 'PO4', 'PO7', 'PO8',
                      'CP3', 'CP5', 'Pz', 'P1', 'P3',
                      'P4', 'P5', 'P6', 'P8', 'AF3',
                      'C4', 'F5', 'Oz', 'O1', 'O2', ]

    bestChannel_29 = ['TP7', 'TP9', 'TP10', 'FC3', 'FC4',
                      'FC5', 'PO7', 'PO8',
                      'CP3', 'CP5', 'P1', 'P3',
                      'P4', 'P5', 'P6', 'P8', 'AF3',
                      'C4', 'F5', 'Oz', 'O1', 'O2',
                      'F4', 'CP2', 'Cz', 'Fp2', 'P2', 'T7', 'FC6']

    bestChannel_32 = ['TP7', 'TP9', 'TP10', 'FC3', 'FC4',
                      'FC5', 'PO3', 'PO4', 'PO7', 'PO8',
                      'CP3', 'CP5', 'Pz', 'P1', 'P3',
                      'P4', 'P5', 'P6', 'P8', 'AF3',
                      'C4', 'F5', 'Oz', 'O1', 'O2',
                      'F4', 'CP2', 'Cz', 'Fp2', 'P2', 'T7', 'FC6']
    # X_train, y_train = loadData(train_rootPath_close)
    # X_test, y_test = loadData(eval_rootPath_close)

    # X_train, y_train = loadData_bestChannel(train_rootPath_close, bestChannel_12)
    # X_test, y_test = loadData_bestChannel(eval_rootPath_close, bestChannel_12)

    X_train_open, y_train_open, X_test_open, y_test_open = combineWptAndIIR_Features(
        wpt_train_path="/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesOpen_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands/train/",
        wpt_eval_path="/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesOpen_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands/eval/",
        iir_train_path="/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesOpen_AllChannels_AllFeatures/train/",
        iir_eval_path="/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesOpen_AllChannels_AllFeatures/eval/",
        bestChannel=bestChannel_29
    )

    X_train_close, y_train_close, X_test_close, y_test_close = combineWptAndIIR_Features(
        wpt_train_path="/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesClose_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands/train/",
        wpt_eval_path="/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesClose_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands/eval/",
        iir_train_path="/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesClose_AllChannels_AllFeatures/train/",
        iir_eval_path="/Users/murphywu/PycharmProjects/eegProject/UNMData/EyesClose_AllChannels_AllFeatures/eval/",
        bestChannel=bestChannel_29
    )
    
    X_train = np.concatenate((X_train_open, X_train_close))
    X_test = np.concatenate((X_test_open, X_test_close))
    y_train = y_train_open + y_train_close
    y_test = y_test_open + y_test_close

    
    X_train = preprocessing.scale(X_train, axis=0)
    X_test = preprocessing.scale(X_test, axis=0)

    # xgBoostParameterOptimizer(X_train, y_train)
    # adaboostParameterOptimizer(X_train, y_train)
    # X_train_close, y_train_close = loadData(train_rootPath_close)
    # X_test_close, y_test_close = loadData(eval_rootPath_close)
    # X_train_open, y_train_open = loadData(train_rootPath_open)
    # X_test_open, y_test_open = loadData(eval_rootPath_open)
    #
    # X_train = np.concatenate((X_train_close, X_train_open))
    # y_train = y_train_close + y_train_open
    # X_test = np.concatenate((X_test_close, X_test_open))
    # y_test = y_test_close + y_test_open
    #
    # 交叉验证的X和y
    X_train = np.concatenate((X_train, X_test))
    y_train = y_train + y_test


    # Iowa test
    X_empty, y_empty, X_test, y_test = combineWptAndIIR_Features(
        wpt_train_path="/Users/murphywu/PycharmProjects/eegProject/IowaData/EyesOpen_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands/test/",
        wpt_eval_path="/Users/murphywu/PycharmProjects/eegProject/IowaData/EyesOpen_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands/test/",
        iir_train_path="/Users/murphywu/PycharmProjects/eegProject/IowaData/EyesOpen_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_IIR/test/",
        iir_eval_path="/Users/murphywu/PycharmProjects/eegProject/IowaData/EyesOpen_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_IIR/test/",
        bestChannel=bestChannel_29
    )


    # 特征选择
    # selector = SelectKBest(f_classif, k=4950)
    # selector.fit(X_train, y_train)
    # resultIndex = selector.get_support()
    # deletIndex = np.where(resultIndex == False)
    # X_train = np.delete(X_train, deletIndex, axis=1)
    # X_test = np.delete(X_test, deletIndex, axis=1)
    # X = np.concatenate((X_train, X_test))
    # y = y_train + y_test

    # 多模型比较：
    # Original
    # models = [('Logit', LogisticRegression(max_iter=5000)),
    #           ('KNN', KNeighborsClassifier()),
    #           ('SVM', SVC(probability=True)),
    #           ('AdaBoost', AdaBoostClassifier(random_state=0)),
    #           ('XGBoost-tree', XGBClassifier(booster='gbtree', random_state=0)),
    #           ('XGBoost-linear', XGBClassifier(booster='gblinear', random_state=0)),
    #           ('RF', RandomForestClassifier(random_state=0))]

    # Optimized
    models = [('Logit', LogisticRegression(penalty='l2', max_iter=5000)),
              ('KNN', KNeighborsClassifier(n_neighbors=21)),
              ('SVM', SVC(C=100, gamma=1e-7, kernel='rbf', probability=True)),
              ('AdaBoost', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_split=4),
                                              n_estimators=179, random_state=0)),
              ('XGBoost-tree', XGBClassifier(
                  scilent=True,
                  learning_rate=0.05,
                  gamma=0.58,
                  n_estimators=140,
                  max_depth=3,
                  min_samples_split=3,
                  objective='binary:logistic',
                  nthread=4,
                  scale_pos_weight=1,
                  seed=0,
                  subsample=0.6,
                  colsample_bytree=0.8,
                  reg_alpha=0.01)),
              ('XGBoost-linear', XGBClassifier(
                  booster='gblinear',
                  scilent=True,
                  learning_rate=0.01,
                  gamma=0.03,
                  n_estimators=140,
                  max_depth=13,
                  min_samples_split=2,
                  objective='binary:logistic',
                  nthread=4,
                  scale_pos_weight=1,
                  seed=0,
                  subsample=0.6,
                  colsample_bytree=0.8,
                  reg_alpha=0
              )),
              ('RF', RandomForestClassifier(random_state=0, n_estimators=220))]

    # models = []
    # c_list = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e+04, 1e+05, 1e+06, 1e+07, 1e+08, 1e+09, 1e+10]
    # gamma_list = [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # for max_depth in range(1, 5):
    #     for min_samples_split in range(15, 25):
    #         models += [
    #             ('AdaboostWithDecisionTree_maxDepth-' + str(max_depth) + '_minSamplesSplit-' + str(min_samples_split),
    #              AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth,
    #                                                        min_samples_split=min_samples_split),
    #                                 n_estimators=180))]

    # tenFold(X, y, fold=3, name='SVM', model=SVC(C=100, gamma=1e-7, kernel='rbf', probability=True),
    #         save_path="./Imagines/" + str(
    #             len(y_train)) + "-fold-ROC_XGBoost_EyesClose_AllChannels_allFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands&IIR.png")

    # 循环训练模型
    results = []
    roc_ = []
    for name, model in models:
        clf = model.fit(X_train, y_train)
        Score = clf.score(X_test, y_test)
        pred_proba = clf.predict_proba(X_test)
        y_prob = pred_proba[:, 1]
        fpr, tpr, roc_auc, Optimal_threshold, optimal_point = ROC(y_test, y_prob)
        Accuracy, Sensitivity, Specificity = calculate_metric(y_test, y_prob, Optimal_threshold)
        result = [Optimal_threshold, Accuracy, Sensitivity, Specificity, roc_auc, Score, name]
        results.append(result)
        roc_.append([fpr, tpr, roc_auc, name])
        print(name + ": " + str(Accuracy))
        print("Optimal threshold: " + str(Optimal_threshold) +
              ", Optimal False Positive rate: " + str(optimal_point[0]) +
              ", Optimal True Positive rate: " + str(optimal_point[1]))

    df_result = pd.DataFrame(results)
    df_result.columns = ["Optimal_threshold", "Accuracy", "Sensitivity", "Specificity", "AUC_ROC", "R2 Score",
                         "Model_name"]
    df_result.to_csv("UNM&Iowa_BestModelsCompare_EyesOpen&Close_29bestChannels_AllFeatures_wpt&IIR.csv")

    # df_roc = pd.DataFrame(roc_)
    # df_roc.columns = ["FPR", "TPR", "ROC_AUC", "Name"]
    # df_roc.to_csv("ROC_compare_EyesClose_12bestChannels_IIR.csv")

    # 绘制多组对比roc曲线
    color = ["darkorange", "navy", "red", "green", "yellow", "pink", "blue"]
    # plt.figure()
    plt.figure(figsize=(10, 10))
    lw = 2
    # plt.plot(roc_[0][0], roc_[0][1], color=color[0], lw=lw, label=roc_[0][3] + ' (AUC = %0.3f)' % roc_[0][2])
    # plt.plot(roc_[1][0], roc_[1][1], color=color[1], lw=lw, label=roc_[1][3] + ' (AUC = %0.3f)' % roc_[1][2])
    # plt.plot(roc_[2][0], roc_[2][1], color=color[2], lw=lw, label=roc_[2][3] + ' (AUC = %0.3f)' % roc_[2][2])
    # plt.plot(roc_[3][0], roc_[3][1], color=color[3], lw=lw, label=roc_[3][3] + ' (AUC = %0.3f)' % roc_[3][2])
    # plt.plot(roc_[4][0], roc_[4][1], color=color[4], lw=lw, label=roc_[4][3] + ' (AUC = %0.3f)' % roc_[4][2])
    # plt.plot(roc_[5][0], roc_[5][1], color=color[5], lw=lw, label=roc_[5][3] + ' (AUC = %0.3f)' % roc_[5][2])
    # plt.plot(roc_[6][0], roc_[6][1], color=color[6], lw=lw, label=roc_[6][3] + ' (AUC = %0.3f)' % roc_[6][2])

    for i in range(len(models)):
        plt.plot(roc_[i][0], roc_[i][1], color=color[i], lw=lw, label=roc_[i][3] + ' (AUC = %0.3f)' % roc_[i][2])
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Curve')
    plt.legend(loc="lower right")
    # plt.savefig(
    #     "./Imagines/ROC_compare_EyesClose_AllChannels_allFeatures_2000clipLength_15segments_56ica-eyeBlink_wptBands_bestParams.png",
    #     dpi=300)
    plt.show()
