from static.model import *

# instantiating built-in object
res = Classifier(RandomForestClassifier(), param_grid={
    'criterion': ['gini'],
    'max_depth': [37],
    'n_estimators': [300],
    'max_features': ['sqrt'],
    'min_samples_split': [30],
    'class_weight': ['balanced'],
    'oob_score': [True]
})
