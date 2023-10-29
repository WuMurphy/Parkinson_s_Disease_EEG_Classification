import os
import shutil
import stat

import numpy as np
from psutil._compat import xrange
from sklearn.model_selection import StratifiedKFold

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath
print(">>>[top layer]:" + rootPath)


def kFolder(kfold, dataSet, dataLabel):
    """
    shuffle:是否在每次分割之前打乱顺序
    :param kfold:
    :param dataSet:
    :param dataLabel:
    :return:
    """
    kfold = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=np.random.seed())

    for n_fold, (tr_idx, val_idx) in enumerate(kfold.split(dataSet, dataLabel)):
        return tr_idx, val_idx


"""
快速的列转化为行
"""


def fast_list2arr(data, offset=None, dtype=None):
    """
    Convert a list of numpy arrays with the same size to a large numpy array.
    This is way more efficient than directly using numpy.array()
    See
        https://github.com/obspy/obspy/wiki/Known-Python-Issues
    :param data: [numpy.array]
    :param offset: array to be subtracted from the each array.
    :param dtype: data type
    :return: numpy.array
    """
    num = len(data)
    out_data = np.empty((num,) + data[0].shape, dtype=dtype if dtype else data[0].dtype)
    for i in xrange(num):
        print(">>>fast_list2arr[index]:" + str(i))
        out_data[i] = data[i] - offset if offset else data[i]
    return out_data


def transDatasetAndLabel(filePath, dataSource=None, fileNameList=None):
    print(">>[start read file]:" + filePath)
    if dataSource is None:
        dataSource = np.load(filePath, allow_pickle=True)

    dataset = []
    dataLabel = []

    for item in dataSource:
        tempDataset = item['data']
        tempLabel = item['label']
        fileName = item['fileName']
        if fileNameList is not None:
            fileNameList.append(fileName)
        assert tempDataset is not None
        assert tempLabel is not None
        dataset.append(tempDataset)
        dataLabel.append(tempLabel)

    del dataSource
    return fast_list2arr(dataset), np.array(dataLabel)


def transforEegData(filePath):
    dataSource = np.load(filePath, allow_pickle=True)
    dataset = []
    dataLabel = []

    for item in dataSource:
        tempDataset = item['data']
        tempLabel = item['label']
        assert tempDataset is not None
        assert tempLabel is not None
        dataset.append(tempDataset)
        dataLabel.append(tempLabel)

    return fast_list2arr(dataset), np.array(dataLabel)


def readEegData(datasetPath, datasetLabelPath):
    dataset = np.load(datasetPath, allow_pickle=True)
    datasetLabel = np.load(datasetLabelPath, allow_pickle=True)
    return dataset, datasetLabel


# filePath:文件夹路径
def delete_file(filePath):
    if os.path.exists(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(filePath)
        return True
    else:
        return False
