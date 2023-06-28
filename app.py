from flask import Flask, request, jsonify

from static.dl import *

app = Flask(__name__)
# app.secret_key = 'Larghetto'

# instantiating built-in object
app.res = clf(RandomForestClassifier(), param_grid={
    'criterion': ['gini'],
    'max_depth': [37],
    'n_estimators': [300],
    'max_features': ['sqrt'],
    'min_samples_split': [30],
    'class_weight': ['balanced'],
    'oob_score': [True]
})
# check if the model has been trained
app.res.is_trained = False


# @app.route('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico',
#                                mimetype='image/vnd.microsoft.icon')


@app.route('/')
def hello():
    return 'hello'


# train path
@app.route('/train', methods=['post'])
def train():
    if 'file' not in request.files:
        return jsonify({'ERROR': 'File not uploaded!'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'ERROR': 'File not uploaded!'})

    sample_id, features, label = data_load(file)
    X_train, y_train, selection_model = train_pre_process(features, label)

    app.res.train(X_train, y_train)
    app.res.is_trained = True
    app.res.selection_model = selection_model

    return 'success'


# test path
@app.route('/test', methods=['post'])
def test():
    if 'file' not in request.files:
        return jsonify({'ERROR': 'File not uploaded!'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'ERROR': 'File not uploaded!'})

    if not app.res.is_trained:
        return jsonify({'ERROR': 'Model not trained!'})

    sample_id_v, features_v, label_v = data_load(file)
    X, y = test_pre_process(features_v, label_v, app.res.selection_model)
    report, matrix = app.res.test(X, y)

    result = {**report, **matrix}

    return jsonify(result)


if __name__ == '__main__':
    app.run()
