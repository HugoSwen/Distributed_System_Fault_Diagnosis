import base64
import io
import json

import pandas as pd
import numpy as np
import sklearn.impute as impute
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as pl


class DLRFC:
    def __init__(self):
        self.selection_model = None
        self.election_model_1 = None
        self.rfc1 = RandomForestClassifier(criterion='gini',
                                           max_depth=17,
                                           n_estimators=208,
                                           max_features='sqrt',
                                           min_samples_split=20,
                                           class_weight='balanced',
                                           random_state=42,
                                           oob_score=True,
                                           min_samples_leaf=10,
                                           n_jobs=-1)

        self.rfc2 = RandomForestClassifier(criterion='gini',
                                           max_depth=17,
                                           n_estimators=200,
                                           max_features='sqrt',
                                           min_samples_split=20,
                                           class_weight='balanced',
                                           random_state=42,
                                           oob_score=True,
                                           min_samples_leaf=10,
                                           n_jobs=-1)

    def data_load(self, path):
        df = pd.read_csv(path)
        sample_id = df['sample_id'].values
        features = df.loc[:, 'feature0':'feature106'].values
        label = df['label'].values
        return sample_id, features, label

    def data_load_test(self, path):
        df = pd.read_csv(path)
        sample_id = df['sample_id'].values
        features = df.loc[:, 'feature0':'feature106'].values
        return sample_id, features

    def train_pre_process(self, X, y, threshold):
        imputer = impute.SimpleImputer(strategy="most_frequent")
        imputer.fit(X)
        X = imputer.transform(X)
        clf = ExtraTreesClassifier(n_estimators=70, n_jobs=-1, random_state=42)
        clf = clf.fit(X, y)
        selection_model = SelectFromModel(clf, prefit=True, threshold=threshold)
        X = selection_model.transform(X)
        return X, y, selection_model

    def valid_pre_process(self, X, selection_model):
        imputer = impute.SimpleImputer(strategy="most_frequent")
        imputer.fit(X)
        X = imputer.transform(X)
        X = selection_model.transform(X)
        return X

    def convert(self):
        pl.show()
        img_io = io.BytesIO()  # io字节流
        pl.savefig(img_io, format='png')  # 将图形保存为 png 格式，并将图像数据写入到 img_io对象中。
        img_io.seek(0)  # 将文件指针移动到图像数据的开头位置
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')  # 将图像流转换为 Base64 编码的字符串
        image = {'image': img_base64}  # 将图像保存为 Base64 编码的字符串字典
        pl.close()  # 清空内存
        return image

    def plot_matrix(self, y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
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

        matrix = self.convert()
        return matrix

    def train(self, path):
        sample_id, features, label = self.data_load(path)
        X_train, y_train, self.selection_model = self.train_pre_process(features, label, '1.5*mean')
        self.rfc1.fit(X_train, y_train)

        # 1 2训练集
        category_filter_1 = np.isin(label, [1, 2])  # 过滤器
        X_filtered_1 = features[category_filter_1]
        y_filtered_1 = label[category_filter_1]
        X_train_1, y_train_1, self.selection_model_1 = self.train_pre_process(X_filtered_1, y_filtered_1, '1.85*mean')
        self.rfc2.fit(X_train_1, y_train_1)

    def validate(self, path):
        sample_id_v, features_v, label_v = self.data_load(path)
        X_test = self.valid_pre_process(features_v, self.selection_model)
        X_test_1 = self.valid_pre_process(features_v, self.selection_model_1)
        y_true = label_v
        y_pred = self.rfc1.predict(X_test)

        for i in range(y_pred.shape[0]):
            if y_pred[i] == 1 or y_pred[i] == 2:
                y_pred[i] = self.rfc2.predict(X_test_1[i].reshape(1, X_test_1.shape[1]))

        # 分类评估报告
        report = metrics.classification_report(y_true, y_pred, output_dict=True)
        # 混淆矩阵
        matrix = self.plot_matrix(y_true, y_pred, [0, 1, 2, 3, 4, 5], title='confusion_matrix',
                                  axis_labels=['0', '1', '2', '3', '4', '5'])
        result = {**report, **matrix}
        return result

    def test(self, path):
        sample_id_t, features_t = self.data_load_test(path)

        X_test = self.valid_pre_process(features_t, self.selection_model)
        X_test_1 = self.valid_pre_process(features_t, self.selection_model_1)
        y_pred = self.rfc1.predict(X_test)

        for i in range(y_pred.shape[0]):
            if y_pred[i] == 1 or y_pred[i] == 2:
                y_pred[i] = self.rfc2.predict(X_test_1[i].reshape(1, X_test_1.shape[1]))
        # 预测结果转字典
        y_pred_dict = {str(index): int(value) for index, value in enumerate(y_pred)}

        unique_labels = set(y_pred)
        counts = [sum(y_pred == label) for label in unique_labels]  # 计算每个分类的样本数量
        labels = [f"Class {label}" for label in unique_labels]  # 创建分类标签的字符串列表
        colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']

        # 绘制分类结果饼状图
        def autopct_format(value):
            return f'{value:.1f}% ({int(value * sum(counts) / 100)})'

        pl.title("Classification Result")
        pl.pie(x=counts, labels=labels, colors=colors, autopct=autopct_format)
        pl.axis('equal')  # 设置纵横比为相等
        pieImage = self.convert()  # 饼状图 Base64 编码字符串json

        result = {**y_pred_dict, **pieImage}  # 合并结果
        return result
