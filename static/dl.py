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

    img_io = io.BytesIO()
    pl.savefig(img_io, format='png')
    img_io.seek(0)

    # 将图像流转换为 Base64 编码的字符串
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # 将图像转换为字典
    matrix = {
        'image': img_base64
    }

    return matrix


class Classifier:
    def __init__(self, module, param_grid):
        self.selection_model = None
        self.param_grid = param_grid
        self.module = module

    def train(self, X, y):
        self.rfc_cv = GridSearchCV(estimator=self.module, param_grid=self.param_grid, scoring='f1_macro', cv=5)
        self.rfc_cv.fit(X, y)
        # for key in self.rfc_cv.best_params_.keys():
        #     print('%s = %s' % (key, self.rfc_cv.best_params_[key]))

    def predict(self, X):
        predict_res = self.rfc_cv.predict(X)
        predict_dict = {str(index): str(value) for index, value in enumerate(predict_res)}
        return predict_dict

    def test(self, X, y):
        predict_res = self.rfc_cv.predict(X)
        # test classification report
        report = metrics.classification_report(predict_res, y, output_dict=True)
        # confusion matrix
        matrix = plot_matrix(y, predict_res, [0, 1, 2, 3, 4, 5], title='confusion_matrix',
                             axis_labels=['0', '1', '2', '3', '4', '5'])
        result = {**report, **matrix}
        return result


