import base64
import io

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import sklearn.impute as impute
import sklearn.metrics as metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def convert():
    pl.show()
    img_io = io.BytesIO()  # io字节流
    pl.savefig(img_io, format='png')  # 将图形保存为 png 格式，并将图像数据写入到 img_io对象中。
    img_io.seek(0)  # 将文件指针移动到图像数据的开头位置
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')  # 将图像流转换为 Base64 编码的字符串
    image = {'image': img_base64}  # 将图像保存为 Base64 编码的字符串字典
    pl.close()  # 清空内存
    return image


def data_load(path):
    df = pd.read_csv(path)
    sample_id = df['sample_id'].values
    features = df.loc[:, 'feature0':'feature106'].values
    label = df['label'].values
    return sample_id, features, label


def data_load_test(path):
    df = pd.read_csv(path)
    sample_id = df['sample_id'].values
    features = df.loc[:, 'feature0':'feature106'].values
    return sample_id, features


def train_pre_process(X, y, threshold):
    imputer = impute.SimpleImputer(strategy="most_frequent")
    imputer.fit(X)
    X = imputer.transform(X)
    clf = ExtraTreesClassifier(n_estimators=70, n_jobs=-1, random_state=42)
    clf = clf.fit(X, y)
    selection_model = SelectFromModel(clf, prefit=True, threshold=threshold)
    X = selection_model.transform(X)
    print('train selected features:' + str(X.shape))
    return X, y, selection_model


def valid_pre_process(X, selection_model):
    imputer = impute.SimpleImputer(strategy="most_frequent")
    imputer.fit(X)
    X = imputer.transform(X)
    X = selection_model.transform(X)
    print('test selected features:' + str(X.shape))
    return X


def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar()

    if title is not None:
        pl.title(title)
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels)
    pl.yticks(num_local, axis_labels)
    pl.ylabel('True label')
    pl.xlabel('Predicted label')

    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")
    # 混淆矩阵Base64编码字符串json
    matrix = convert()
    return matrix


def validate_result(y, y_pred):
    # 分类评估报告
    report = metrics.classification_report(y, y_pred, output_dict=True)
    # 混淆矩阵
    matrix = plot_matrix(y, y_pred, [0, 1, 2, 3, 4, 5], title='confusion_matrix',
                         axis_labels=['0', '1', '2', '3', '4', '5'])
    result = {**report, **matrix}
    return result
