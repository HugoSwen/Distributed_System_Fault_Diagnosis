import base64
import io

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import sklearn.impute as impute
import sklearn.metrics as metrics
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

torch.cuda.is_available()


def convert():
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


def train_pre_process(X, y):
    Impute = impute.SimpleImputer(strategy="most_frequent")
    Impute.fit(X)
    X = Impute.transform(X)

    classifier = ExtraTreesClassifier(n_estimators=50)
    classifier = classifier.fit(X, y)
    selection_model = SelectFromModel(classifier, prefit=True)

    X = selection_model.transform(X)
    # print(X.shape)
    return X, y, selection_model


def test_pre_process(X, y, selection_model):
    Impute = impute.SimpleImputer(strategy="most_frequent")
    Impute.fit(X)
    X = Impute.transform(X)

    X = selection_model.transform(X)
    # print(X.shape)
    return X, y


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


class Classifier:
    # 类初始化
    def __init__(self, module, param_grid):
        self.selection_model = None
        self.param_grid = param_grid
        self.module = module

    def train(self, X, y):
        self.rfc_cv = GridSearchCV(estimator=self.module, param_grid=self.param_grid, scoring='f1_macro',
                                   cv=5)  # 5折交叉验证
        self.rfc_cv.fit(X, y)

    def predict(self, X):
        y_pred = self.rfc_cv.predict(X)
        # 预测结果转字典
        y_pred_dict = {str(index): int(value) for index, value in enumerate(y_pred)}
        # 绘制分类结果柱状图
        unique_labels = set(y_pred)
        counts = [sum(y_pred == label) for label in unique_labels]  # 计算每个分类的样本数量
        labels = [f"Class {label}" for label in unique_labels]  # 创建分类标签的字符串列表
        bars = pl.bar(labels, counts)  # 创建BarContainer对象

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            pl.text(bar.get_x() + bar.get_width() / 2, height, count,
                    ha='center', va='bottom')

        pl.title("Classification Result")
        pl.xlabel("Class")
        pl.ylabel("Count")
        # 柱状图 Base64 编码字符串json
        barImage = convert()
        # 合并结果
        result = {**y_pred_dict, **barImage}

        return result

    def test(self, X, y):
        y_pred = self.rfc_cv.predict(X)
        # 分类评估报告
        report = metrics.classification_report(y, y_pred, output_dict=True)
        # 混淆矩阵
        matrix = plot_matrix(y, y_pred, [0, 1, 2, 3, 4, 5], title='confusion_matrix',
                             axis_labels=['0', '1', '2', '3', '4', '5'])
        # 合并结果
        result = {**report, **matrix}

        return result
