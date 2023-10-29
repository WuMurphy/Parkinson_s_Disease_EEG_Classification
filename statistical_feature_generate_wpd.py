import gc
import os

import sys

sys.path
sys.path.append('')

import time
from pathlib import Path
import EntropyHub as EH

import pywt

import numpy as np

import resampy

from commonUtils import rootPath
# from dataGenerate.utils.loadEdf import LoadEdf, get_all_sorted_file_names
from logger_config import logger


# 样本熵计算
def sampleEntropy2(Datalist, r, m=2):
    th = r * np.std(Datalist)  # 容限阈值
    return EH.SampEn(Datalist, m, r=th)[0][-1]


def segmentSignals(data, clipLength=2000, initSelectSegment=100):
    """
    分片截取数据
    :param data:
    :param clipLength: 默认8s数据(250Hz采样率)
    :param initSelectSegment: 默认分100段
    :return: segmentList
    """
    dataLength = len(data)
    if dataLength < clipLength:
        raise Exception('the data length is smaller than the clipLength')

    segmentList = []
    startIndex = 0
    segmentNum = 0
    while True:
        endIndex = startIndex + clipLength
        if endIndex > dataLength:
            break

        item = data[startIndex:endIndex]
        segmentList.append(item)
        startIndex = startIndex + clipLength  # start = end: no overlapping

        segmentNum = segmentNum + 1
        if segmentNum == initSelectSegment:
            break

    return segmentList


def wpdForSegment(data, wavelet='db4', maxlevel=8, channelName=None):
    """
    小波包分解
    :param data:
    :param wavelet:
    :param maxlevel:
    :param startLevel:
    :param label:
    :return:
    """

    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    abs_mean_List = []
    featureAggreate = []
    wpdCoefficientLabel = ['aaaaaa', 'aaaaad', 'aaaada', 'aaaadd', 'aaaad', 'aaad']

    for i in range(len(wpdCoefficientLabel)):
        freqTreeLabel = wpdCoefficientLabel[i]
        currentData = wp[freqTreeLabel]
        reconstructData = currentData.reconstruct(update=False)
        currentData = currentData.data

        feature = computeFeature(currentData)
        feature['coefficientLabel'] = freqTreeLabel

        abs_mean_List.append(feature['abs_mean'])

        featureAggreate.append(feature)

    def rmavFeature(mavList):
        # ref: https://github.com/Eldave93/Seizure-Detection-Tutorials/blob/master/02.%20Pre-Processing%20%26%20Feature%20Engineering.ipynb
        feature = []
        for level_no in range(0, len(mavList)):
            # for the first decimation
            if level_no == 0:
                item = mavList[level_no] / mavList[level_no + 1]
                feature.append(item)
                continue

            # for the last decimation
            elif level_no == len(mavList) - 1:
                item = mavList[level_no] / mavList[level_no - 1]
                feature.append(item)
            else:
                before = mavList[level_no - 1]
                after = mavList[level_no + 1]
                mean_data = (before + after) / 2
                item = mavList[level_no] / mean_data
                feature.append(item)

        return feature

    rmav_feature = rmavFeature(abs_mean_List)

    for index in range(0, len(rmav_feature)):
        featureAggreate[index]['rmav'] = rmav_feature[index]
        featureAggreate[index]['channelName'] = channelName

    return featureAggreate


def computeFeature_signal(reconstructData, currentFeature):
    # TODO: add more features for the reconstructData, include: Autoregressive model parameters, Sample entropy,
    #  Hurst exponent and Hjorth parameters for each band
    pass


def computeFeature(currentData):
    abs_mean = np.mean(np.abs(currentData))
    mean = np.mean(currentData)
    power_mean = np.mean(np.power(currentData, 2))
    sd = np.std(currentData)

    features = {}
    features['abs_mean'] = abs_mean
    features['mean'] = mean
    features['power_mean'] = power_mean
    features['sd'] = sd
    # 调用 kurtosis 函数时指定不使用Fisher的定义，使用Pearson的定义，详细参数见 scipy.stats.kurtosis  bias参数也要注意
    # matlab  kurtosis(testData,0)   skewness(testData,0)  值一致
    # skew1 = stats.skew(currentData, bias=False)
    # kurt1 = stats.kurtosis(currentData, fisher=False, bias=False)

    import pandas as pd
    s = pd.Series(currentData)
    skew = s.skew()
    kurt = s.kurt()

    features['skew'] = skew
    features['kurt'] = kurt

    mad = np.mean(np.abs(currentData - mean))

    percentile = np.percentile(currentData, (25, 75))
    iqr = percentile[1] - percentile[0]

    features['mad'] = mad
    features['iqr'] = iqr

    # 频域特征提取
    data_fft = np.fft.fft(currentData)
    N = data_fft.shape[0]  # 样本个数 和 信号长度

    # 傅里叶变换是对称的，只需取前半部分数据，否则由于 频率序列 是 正负对称的，会导致计算 重心频率求和 等时正负抵消
    mag = np.abs(data_fft)[: N // 2]  # 信号幅值
    freq = np.fft.fftfreq(N, 1 / 500)[: N // 2]
    # mag = np.abs(data_fft)[: , N // 2: ]  # 信号幅值
    # freq = np.fft.fftfreq(N, 1 / sampling_frequency)[N // 2: ]

    ps = mag ** 2 / N  # 功率谱

    features['fc'] = np.sum(freq * ps) / np.sum(ps)  # 重心频率
    features['mf'] = np.mean(ps)  # 平均频率
    features['rmsf'] = np.sqrt(np.sum(ps * np.square(freq)) / np.sum(ps))  # 均方根频率

    # 熵特征提取
    features['se'] = sampleEntropy2(currentData, r=0.2, m=2)  # 样本熵

    return features


def doComputeCLBP(data, clipLength=2000, initSelectSegment=100, wavelet='db4', maxlevel=5, channelName=None):
    segmentList = segmentSignals(data, clipLength=clipLength, initSelectSegment=initSelectSegment)
    concatData = []

    for segment in segmentList:
        wcList = wpdForSegment(segment, wavelet=wavelet, maxlevel=maxlevel, channelName=channelName)
        concatData.append(wcList)

    concatData = np.array(concatData)
    return concatData


'''
def loadData(sourceRootPath, trainLabel, preproc_functions=None,
             initSelectSegment=100,
             resampleRate=250,
             clipLength=250 * 8,
             wavelet='db4',     # selected wavelet
             maxlevel=8,
             strftime=None):
    gc.collect()

    if strftime is not None:
        strftime = strftime % (wavelet, resampleRate, int(clipLength / resampleRate), initSelectSegment)
        parentPath = rootPath + '/dataGenerate/wpdDataSource/' + strftime + "/"
        if not os.path.exists(parentPath):
            os.makedirs(parentPath)

    parentPath = rootPath + '/dataGenerate/wpdDataSource/' + strftime + "/" + trainLabel + "/"
    if not os.path.exists(parentPath):
        os.makedirs(parentPath)

    edf = LoadEdf(montage_file_to_use=rootPath + '/dataGenerate/utils/defaultsMontage/gbdt.txt')
    all_file_names = get_all_sorted_file_names(trainLabel, sourceRootPath)
    fileSize = len(all_file_names)
    for index, fname in enumerate(all_file_names):
        (path, filename) = os.path.split(fname)
        savePath = parentPath + str(index + 1) + "_" + filename + '.npy'

        # if the file path exists, which means this data has been processed, then skip it to save time
        # command it whenever you want a new processed dataset.
        # if os.path.exists(savePath):
        #     continue

        logger.warning("Load {:d} of {:d}".format(index + 1, fileSize))
        edf.open_edf_file(fname, ignore_head=True, ignore_montage=True, ignore_annotations=False)
        rawChannelData = edf.raw_channels
        # 选择需要处理的通道
        rawChannelData = edf.transforData(rawChannelData)
        orginSampleRate = edf.sampling_rate

        age = edf.header['ltpi_age']
        if age is not "" and age is not None:
            age = age.split(':')[1]
            age = np.float(age)

        gender = edf.header['ltpi_gender']

        channelKeys = list(rawChannelData.keys())
        newFeatureSet = []
        for channelName in rawChannelData:
            data = rawChannelData[channelName]

            for fn in preproc_functions:
                data, orginSampleRate = fn(data, orginSampleRate)

            if orginSampleRate != resampleRate:
                data = resampy.resample(data, orginSampleRate, resampleRate, axis=0,
                                        filter='kaiser_fast')
                orginSampleRate = resampleRate

            dataItem = data.astype(np.float32)
            dataItem = doComputeCLBP(dataItem, clipLength=clipLength, initSelectSegment=initSelectSegment,
                                     wavelet=wavelet, maxlevel=maxlevel, channelName=channelName)
            newFeatureSet.append(dataItem)

        rawChannelData = np.array(newFeatureSet)
        assert rawChannelData is not None

        if '/abnormal/' in Path(fname).as_posix():
            findex_ = 1
        else:
            findex_ = 0

        data = {
            "label": findex_,
            "fileName": fname,
            "axis": 'abnormal' if findex_ == 1 else 'normal',
            'source': rawChannelData,
            "channelKeys": channelKeys,
            "sampleFrequency": orginSampleRate,
            "clipLength": clipLength,
            "initSelectSegment": initSelectSegment,
            "wavelet": wavelet,
            "maxLevel": maxlevel,
            "age": age,
            "gender": gender
        }

        (path, filename) = os.path.split(fname)
        timestamp = time.time()
        timestruct = time.localtime(timestamp)

        np.save(savePath, data)

'''
if __name__ == '__main__':

    wpdCoefficientLabel = []
    for i in range(1, 9):
        wpdCoefficientLabel.append('a' * i)
        wpdCoefficientLabel.append('d' * i)

    # how many minutes to use per recording
    duration_recording_mins = 15

    preproc_functions = []
    preproc_functions.append(
        lambda data, fs: (data[:int(duration_recording_mins * 60 * fs)], fs))

    timestamp = time.time()
    timestruct = time.localtime(timestamp)
    strftime = time.strftime('%Y%m%d_%H_%M_%S', timestruct)

    data_folders = ['C:/Users/haoyu.wu18/Desktop/EEG demo/EEG demo/v2.0.0/edf']

    dataSetLabel = ['train', 'eval']
    # for dataSetLabelItem in dataSetLabel:
    #     loadData(data_folders, trainLabel=dataSetLabelItem,
    #              preproc_functions=preproc_functions,
    #              resampleRate=250,
    #              clipLength=250 * 8,
    #              wavelet='sym4',
    #              maxlevel=8,
    #              initSelectSegment=100,
    #              strftime=strftime + "_wpd_%s_%shz_%smin_%ssegments")
