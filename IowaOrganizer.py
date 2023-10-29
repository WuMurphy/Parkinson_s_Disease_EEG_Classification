import os
import threading, queue
import time
import traceback
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import mne
import pywt
import resampy

from scipy.io import loadmat
from statistical_feature_generate_wpd import segmentSignals, computeFeature, wpdForSegment
from sklearn import preprocessing
from mne.preprocessing import ICA
from mne_icalabel import label_components

from logger_config import logger

# # UNM sample rates: 500Hz
# UNM_parent_path = "C:/Users/haoyu.wu18/Desktop/Data and Code/Data and Code/Dataset/UNMDataset/"
#
#
# # 这里的HDF5 object reference里存的是一个reference，就是地址，最终还是要通过根，就是这里一开始读取的mat，来查找对应位置的数据，即mat[reference]
# # 那么对于嵌套的结构，也需要反复获取下一层里的reference直到找到最底层拿到数据。
# mat = h5py.File(UNM_parent_path + "Organized_data/EEG_Jim_rest_Unsegmented_WithAllChannelsEYESOPEN.mat")
# print(mat.keys())
# location_ref = mat['Channel_location']
# location = np.array([mat[location_ref[i, 0]] for i in range(location_ref.size)])
# eeg = mat['EEG']
# cell_0 = mat[eeg[0, 0]]
# cell_0_0 = mat[cell_0[0, 0]]
# cell_0_0_0 = mat[cell_0_0[0, 0]]
# data = np.array(cell_0_0_0).T
# pass

class myThread(threading.Thread):
    def __init__(self, name, q, person_list, label, label_num, file_type, clipLength, initSelectSegment,
                 source_root_path, save_root, list_PD, list_HC, bandDict,
                 save_root_path, index, resampleRate, dataType):
        threading.Thread.__init__(self)
        self.name = name
        self.q = q

        # loadData的参数
        self.person_list = person_list
        self.label = label
        self.label_num = label_num
        self.file_type = file_type
        self.clipLength = clipLength
        self.initSelectSegment = initSelectSegment
        self.source_root_path = source_root_path
        self.save_root = save_root
        self.list_PD = list_PD
        self.list_HC = list_HC
        self.bandDict = bandDict
        self.save_root_path = save_root_path
        self.index = index
        self.resampleRate = resampleRate
        self.dataType = dataType

    def run(self):
        print("Starting " + self.name)
        while True:
            try:
                # start = time.time()
                loadPerson(self.name, self.q, self.person_list, self.label, self.label_num, self.file_type,
                           self.clipLength, self.initSelectSegment, self.source_root_path, self.save_root,
                           self.list_PD, self.list_HC, self.bandDict, self.save_root_path,
                           self.index, self.resampleRate, self.dataType)
                # end = time.time()
                # print(end-start)
                # pass
            except:
                break
        print("Exiting " + self.name)


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


def bandPassChannel(data, bandDict, channelName=None, sfreq=500):
    ch_types = ['eeg', 'eeg']
    dataSqueeze = np.squeeze(np.array([data, data]))
    info = mne.create_info(ch_names=[channelName, 'None'], sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(dataSqueeze, info)

    data_bandPassed = []
    for bandLabel in bandDict.keys():
        bandRange = bandDict[bandLabel]
        l_freq, h_freq = bandRange[0], bandRange[1]
        bandData = raw.load_data().filter(l_freq=l_freq, h_freq=h_freq, method="iir")[channelName][0][0, :]
        data_bandPassed.append(bandData)

    return np.array(data_bandPassed)


def segmentBands(data, clipLength=2000, initSelectSegment=15):
    dataLength = data.shape[1]
    if dataLength < clipLength:
        raise Exception('the data length is smaller than the clipLength')

    segmentList = []
    startIndex = 0
    segmentNum = 0
    while True:
        endIndex = startIndex + clipLength
        if endIndex > dataLength:
            break

        item = data[:, startIndex:endIndex]
        segmentList.append(item)
        startIndex = startIndex + clipLength  # start = end: no overlapping

        segmentNum = segmentNum + 1
        if segmentNum == initSelectSegment:
            break

    return segmentList


def featureForSegment(segment, bandDict, channelName=None):
    abs_mean_List = []
    featureAggreate = []
    bandIndex = 0
    for bandLabel in bandDict.keys():
        bandData = segment[bandIndex, :]
        feature = computeFeature(bandData)
        feature['bandLabel'] = bandLabel
        abs_mean_List.append(feature['abs_mean'])

        featureAggreate.append(feature)
        bandIndex += 1

    rmav_feature = rmavFeature(abs_mean_List)

    for index in range(0, len(rmav_feature)):
        featureAggreate[index]['rmav'] = rmav_feature[index]
        featureAggreate[index]['channelName'] = channelName

    return featureAggreate


def bandPassForSegment(segment, bandDict, channelName=None, sfreq=500):
    ch_types = ['eeg', 'eeg']
    dataSqueeze = np.squeeze(np.array([segment, segment]))
    info = mne.create_info(ch_names=[channelName, 'None'], sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(dataSqueeze, info)

    abs_mean_List = []
    featureAggreate = []
    for bandLabel in bandDict.keys():
        bandRange = bandDict[bandLabel]
        l_freq, h_freq = bandRange[0], bandRange[1]
        bandData = raw.load_data().filter(l_freq=l_freq, h_freq=h_freq, filter_length=len(segment)-1, method="iir")[channelName][0][0, :]
        feature = computeFeature(bandData)
        feature['bandLabel'] = bandLabel
        abs_mean_List.append(feature['abs_mean'])

        featureAggreate.append(feature)

    # def rmavFeature(mavList):
    #     # ref: https://github.com/Eldave93/Seizure-Detection-Tutorials/blob/master/02.%20Pre-Processing%20%26%20Feature%20Engineering.ipynb
    #     feature = []
    #     for level_no in range(0, len(mavList)):
    #         # for the first decimation
    #         if level_no == 0:
    #             item = mavList[level_no] / mavList[level_no + 1]
    #             feature.append(item)
    #             continue
    #
    #         # for the last decimation
    #         elif level_no == len(mavList) - 1:
    #             item = mavList[level_no] / mavList[level_no - 1]
    #             feature.append(item)
    #         else:
    #             before = mavList[level_no - 1]
    #             after = mavList[level_no + 1]
    #             mean_data = (before + after) / 2
    #             item = mavList[level_no] / mean_data
    #             feature.append(item)
    #
    #     return feature

    rmav_feature = rmavFeature(abs_mean_List)

    for index in range(0, len(rmav_feature)):
        featureAggreate[index]['rmav'] = rmav_feature[index]
        featureAggreate[index]['channelName'] = channelName

    return featureAggreate


def doComputeCLBP_My(data, clipLength=2000, initSelectSegment=100, bandDict={}, channelName=None, sfreq=500):
    """
    this function will firstly use IIR bandpass filter to generate 6 bands and then divide them into several segments
    and extract features from each band in every segment, rather than segmenting first and applying IIR for each segment.
    :param data:
    :param clipLength:
    :param initSelectSegment:
    :param bandDict:
    :param channelName:
    :param sfreq:
    :return:
    """
    data_bandPassed = bandPassChannel(data, bandDict, channelName, sfreq)
    segmentList = segmentBands(data_bandPassed, clipLength, initSelectSegment)
    concatData = []

    for segment in segmentList:
        feature_segment = featureForSegment(segment, bandDict, channelName)
        concatData.append(feature_segment)

    concatData = np.array(concatData)
    return concatData


def doComputeCLBP(data, clipLength=2000, initSelectSegment=100, bandDict={}, channelName=None, sfreq=500):
    segmentList = segmentSignals(data, clipLength=clipLength, initSelectSegment=initSelectSegment)
    concatData = []

    for segment in segmentList:
        # bpList = wpdForSegment(segment, wavelet='db5', maxlevel=6, channelName=channelName)

        bpList = bandPassForSegment(segment, bandDict, channelName=channelName, sfreq=sfreq)
        concatData.append(bpList)

    concatData = np.array(concatData)
    return concatData


def preprocessingData(data, rawChannels, sfreq, n_components):
    """
    To be completed... ...
    :param data:
    :param rawChannels:
    :param sfreq:
    :param icaComponents:
    :return:
    """
    ch_types = ['eeg' for i in range(len(rawChannels))]
    # ch_types[-1] = 'eog'
    dataSqueeze = np.squeeze(np.array([data[i, :] for i in range(data.shape[0])]))
    info = mne.create_info(ch_names=rawChannels, sfreq=sfreq, ch_types=ch_types)
    rawData = mne.io.RawArray(dataSqueeze, info)
    # rawData.plot()
    iowa_montage = mne.channels.make_standard_montage('brainproducts-RNP-BA-128')
    rawData.set_montage(iowa_montage)
    # rawData_copy = rawData.copy()

    rawData.load_data().filter(l_freq=1., h_freq=100.0)
    rawData.load_data().notch_filter(freqs=60.)
    rawData = rawData.set_eeg_reference("average")
    # rawData_copy.plot()
    # dataSqueeze = np.squeeze(np.insert(np.array([rawData_copy[rawChannels[i]][0][0, :] for i in range(len(rawChannels)-1)]), -1, rawData['VEOG'][0][0, :], axis=0))
    # rawData_temp = mne.io.RawArray(dataSqueeze, info)

    ica = ICA(n_components=n_components,
              max_iter="auto",
              method="infomax",
              random_state=97,
              fit_params=dict(extended=True))
    # ica.noise_cov = mne.compute_raw_covariance(rawData_copy)
    ica.fit(rawData)
    # ica.plot_sources(rawData, show_scrollbars=False, show=True)

    # Exclude components except brain and others

    ic_labels = label_components(rawData, ica, method="iclabel")
    labels = ic_labels["labels"]
    # eyeBlink = [i for i in range(len(ic_labels)) if ic_labels[i] == "eye blink"]
    # ica.plot_properties(rawData, picks=eyeBlink, verbose=False, show=True)
    exclude_idx = [idx for idx, label in enumerate(labels) if label in ["eye blink"]]
    # rawData_copy = rawData.copy()
    ica.apply(rawData, exclude=exclude_idx)
    # rawData_copy.plot(show_scrollbars=False)
    # rawData.plot(show_scrollbars=False)
    # del rawData

    # eog_indices = []
    # eog_indices, eog_scores = ica.find_bads_eog(rawData, measure="correlation")
    # ica.exclude = [5, 8, 9, 12, 14, 16]
    # reconst_rawData = rawData.copy()
    # ica.apply(reconst_rawData)
    # rawData.plot()
    # reconst_rawData.plot()

    # ica.plot_components()
    # ica.plot_overlay(rawData, exclude=[16])

    rawData = rawData[:][0]
    return rawData


def drawNoise(data, rawChannels, sfreq=500):
    ch_types = ['eeg' for i in range(len(rawChannels))]
    ch_types[-1] = 'eog'
    dataSqueeze = np.squeeze(np.array([data[i, :] for i in range(data.shape[0])]))
    info = mne.create_info(ch_names=rawChannels, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(dataSqueeze, info)
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage)
    eeg_channels = mne.pick_types(raw.info, eeg=True, eog=True)
    # raw.plot(duration=60, order=eeg_channels, n_channels=len(eeg_channels),
    #          remove_dc=False)
    raw.load_data().filter(l_freq=1., h_freq=40.0)

    # Low-frequency drifts excluded
    # raw.plot(duration=60, order=eeg_channels, n_channels=len(eeg_channels),
    #          remove_dc=False)

    # Power line noise: 60Hz and 180Hz
    # fig = raw.plot_psd(tmax=np.inf, fmax=250, average=True)
    # for ax in fig.axes[0:]:
    #     freqs = ax.lines[-1].get_xdata()
    #     psds = ax.lines[-1].get_ydata()
    #     for freq in (60, 180):
    #         idx = np.searchsorted(freqs, freq)
    #         ax.arrow(x=freqs[idx], y=psds[idx] + 18, dx=0, dy=-12, color='red',
    #                  width=0.1, head_width=3, length_includes_head=True)
    # fig.show()

    # EOG
    eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    eog_epochs.plot_image(combine='mean')
    eog_epochs.average().plot_joint()


def drawRawData(data, rawChannels, channelsToDraw, sfreq):
    ch_types = ['eeg' for i in range(len(channelsToDraw))]
    channelIndex = [np.where(rawChannels == channelsToDraw[i]) for i in range(len(channelsToDraw))]

    dataToDraw = np.squeeze(np.array([data[index, :] for index in channelIndex]))
    info = mne.create_info(ch_names=channelsToDraw, sfreq=sfreq, ch_types=ch_types)
    dataStruc = mne.io.RawArray(dataToDraw, info)
    dataStruc.load_data().filter(l_freq=1., h_freq=None)
    dataStruc.plot()


def findEyeCloseAndOpen(event, sampleRate):
    """
    Find the eyes close and open signals
    :parameter event: the event dataframe contains different states and their time stamps.
                S 1 and S 2 represent the eyes open state and S 3 and S 4 represent the eyes close state
    :parameter sampleRate: the sample rate of the raw EEG data
    :return close_start, close_end, open_start, open_end: the start and end time stamps of the
            eyes close signal and eyes open signal
    """
    close_flag, open_flag = 0, 0
    close_start, close_end, open_start, open_end = 0, 0, 0, 0
    for i in range(1, len(event['type'])):

        if event['type'][i][0] == 'S  3' or event['type'][i][0] == 'S  4':
            close_start = event['latency'][i][0, 0] if close_flag == 0 else close_start
            close_end = event['latency'][i][0, 0] + sampleRate
            close_flag = 1 if close_flag == 0 else close_flag

        elif event['type'][i][0] == 'S  1' or event['type'][i][0] == 'S  2':
            open_start = event['latency'][i][0, 0] if open_flag == 0 else open_start
            open_end = event['latency'][i][0, 0] + sampleRate
            open_flag = 1 if open_flag == 0 else open_flag

    return close_start, close_end, open_start, open_end


def dataStandardization(data):
    """
    Considering the difference of scales between EEG, EOG and accelerometer data, standardization is necessary
    :param data:
    :return:
    """
    max_eeg = np.max(np.abs(data[:data.shape[0]-1, :]))
    max_eog = np.max(np.abs(data[-1, :]))
    scale_ratio = max_eeg / max_eog
    data[-1, :] = data[-1, :] * scale_ratio
    # max_abs_scaler = preprocessing.MaxAbsScaler()
    # return max_abs_scaler.fit_transform(data)

    return preprocessing.normalize(data, axis=1)


def loadPerson(threadName, q, person_list, label, label_num, file_type, clipLength, initSelectSegment, source_root_path,
               save_root, list_PD, list_HC, bandDict, save_root_path, index, resampleRate=500, dataType="EyesClose"):
    person = q.get(timeout=1)

    trainLabel = 'test'
    save_parent_path = save_root_path + trainLabel + "/"
    if not os.path.exists(save_parent_path):
        os.makedirs(save_parent_path)
    savePath = save_parent_path + str(person) + "_" + label + ".npy"
    if not os.path.exists(savePath):
        try:
            index = index + 1
            logger.warning("Load {:d} of {:d}".format(index, 28))

            filename = source_root_path + label + "/" + label+str(person) + ".mat"
            person_data = loadmat(filename)['EEG']
            originalSampleRate = np.int32(person_data['srate'][0, 0][0, 0])

            raw_data = person_data['data'][0, 0][0:63, :]
            # raw_data = dataStandardization(raw_data)

            chanlocs = pd.DataFrame(person_data['chanlocs'][0, 0][0])
            rawChannel_name = [chanlocs['labels'][i][0] for i in range(63)]

            # Data preprocessing
            data = preprocessingData(raw_data, rawChannel_name, originalSampleRate, n_components=56)

            # Feature extraction
            newFeatureSet = []
            for i in range(len(rawChannel_name)):
                print("i = " + str(i))
                # start = time.time()
                dataItem = data[i, :].astype(np.float32)
                # dataItem = eyesOpen_data[i, :].astype(np.float32)

                if originalSampleRate != resampleRate:
                    data = resampy.resample(dataItem, originalSampleRate, resampleRate, axis=0,
                                            filter='kaiser_fast')
                    originalSampleRate = resampleRate

                    dataItem = data.astype(np.float32)

                dataItem = doComputeCLBP(dataItem, clipLength=clipLength, initSelectSegment=initSelectSegment,
                                         bandDict=bandDict, channelName=rawChannel_name[i], sfreq=originalSampleRate)

                newFeatureSet.append(dataItem)
                # end = time.time()
                # print("##################################   " + str(end-start) + "   ##################################")

            source = np.array(newFeatureSet)
            assert source is not None

            # eyesOpen_data = np.array(newFeatureSet)
            # assert eyesOpen_data is not None

            # One person one data
            data = {
                "label": label_num,
                "fileName": filename,
                "axis": label,
                'source': source,
                "channelKeys": rawChannel_name,
                "sampleFrequency": originalSampleRate,
                "clipLength": clipLength,
                "initSelectSegment": initSelectSegment,
            }

            # One person two datas
            # dividNum = initSelectSegment // 2
            # data1 = {
            #     "label": label_num,
            #     "fileName": filename,
            #     "axis": label,
            #     'source': eyesOpen_data[:, 0:dividNum],
            #     "channelKeys": rawChannel_name,
            #     "sampleFrequency": originalSampleRate,
            #     "clipLength": clipLength,
            #     "initSelectSegment": initSelectSegment,
            # }
            #
            # data2 = {
            #     "label": label_num,
            #     "fileName": filename,
            #     "axis": label,
            #     'source': eyesOpen_data[:, dividNum:initSelectSegment],
            #     "channelKeys": rawChannel_name,
            #     "sampleFrequency": originalSampleRate,
            #     "clipLength": clipLength,
            #     "initSelectSegment": initSelectSegment,
            # }


            # savePath1 = save_parent_path + str(person) + "_" + label + "_1.npy"
            # savePath2 = save_parent_path + str(person) + "_" + label + "_2.npy"

            np.save(savePath, data)
            # np.save(savePath1, data1)
            # np.save(savePath2, data2)
        except Exception as e:
            traceback.print_exc()
            print(threadName, "Error: ", e)


def loadData(file_type, clipLength, initSelectSegment, source_root_path, save_root, resampleRate=500, dataType="EyesClose", threadNum=10):
    # 14 PD patients
    list_PD = [1001, 1021, 1031, 1061, 1091, 1101, 1151, 1201, 1251, 1261, 1311, 1571, 1661, 1681]

    # 14 health control
    list_HC = [1021, 1041, 1061, 1081, 1101, 1111, 1191, 1201, 1211, 1231, 1291, 1351, 1381, 1411]

    # Band Dictionary
    bandDict = {'delta': [1.0, 4.0], 'theta': [4.0, 8.0], 'alpha1': [8.0, 10.0], 'alpha2': [10.0, 13.0],
                'alpha': [8.0, 13.0], 'beta': [13.0, 30.0]}

    save_root_path = save_root + file_type + "/"
    index = 0

    for label_num in [0, 1]:
        (label, person_list) = ("PD", list_PD) if label_num == 0 else ("Control", list_HC)

        # 分配线程数
        ThreadNum = threadNum

        # 创建线程共享的受试者队列，避免线程冲突
        workQueue = queue.Queue(len(person_list))
        for person in person_list:
            workQueue.put(person)

        threads = []

        # create 8 threads
        for i in range(1, ThreadNum + 1):
            # This was used for showing the preprocessing ica result, please command it if you don't need that
            # loadPerson("Thread-" + str(i), workQueue, person_list, label, label_num, file_type, clipLength,
            #          initSelectSegment, source_root_path, save_root, list_PD, list_HC, ON_OFF_indicator,
            #          bandDict, save_root_path, index, resampleRate=resampleRate, dataType=dataType)

            # create a new thread
            thread = myThread("Thread-" + str(i), workQueue, person_list, label, label_num, file_type, clipLength,
                              initSelectSegment, source_root_path, save_root, list_PD, list_HC,
                              bandDict, save_root_path, index, resampleRate=resampleRate, dataType=dataType)
            # start the thread
            thread.start()
            # add new thread to the thread list
            threads.append(thread)

        # waiting for all threads finish
        for t in threads:
            t.join()


        # for person in person_list:
        #     loadPerson(person, person_list, label, label_num, file_type, clipLength, initSelectSegment,
        #                source_root_path, save_root, list_PD, list_HC, ON_OFF_indicator, bandDict, save_root_path, index)

    pass

# source_root_path = "/Users/murphywu/PycharmProjects/eegProject/UNMData/Jim_rest/"
# save_root = "/Users/murphzywu/PycharmProjects/eegProject/UNMData/"
#
# loadData("EyesClose_AllChannels_AllFeatures", 2000, 15, source_root_path, save_root)


source_root_path = "/Users/murphywu/PhD/Codes/Data and Code/Dataset/IowaDataset/Imported data/"
save_root = "/Users/murphywu/PycharmProjects/eegProject/IowaData/"

clipLength = 2000
initSelectSegment = 15
dataType = "EyesOpen"

loadData(dataType+"_AllChannels_AllFeatures_2000clipLength_15segments_56ica-eyeBlink_IIR", clipLength,
         initSelectSegment, source_root_path, save_root, dataType=dataType, threadNum=5)
print("############ Program Finish ############")
