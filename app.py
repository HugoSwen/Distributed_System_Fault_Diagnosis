import json

from flask import Flask, request, jsonify, Response

from static.Clf_object import *

app = Flask(__name__)


def checkFile():
    if 'file' not in request.files or request.files['file'].filename == '':
        return False
    return True


def checkModel():
    if res.selection_model is None:
        return False
    return True


@app.route('/')
def hello():
    return 'hello'


# train path
@app.route('/train', methods=['post'])
def train():
    if not checkFile():
        return jsonify({'ERROR': 'File not uploaded!'})

    file = request.files['file']
    sample_id, features, label = data_load(file)
    X_train, y_train, selection_model = train_pre_process(features, label)

    res.train(X_train, y_train)
    res.selection_model = selection_model

    return 'success'


# predict path
@app.route('/predict', methods=['post'])
def predict():
    if not checkFile():
        return jsonify({'ERROR': 'File not uploaded!'})
    if not checkModel():
        return jsonify({'ERROR': 'Model not trained!'})

    file = request.files['file']
    sample_id_v, features_v, label_v = data_load(file)
    X, y = test_pre_process(features_v, label_v, res.selection_model)
    predict_dict = res.predict(X)

    json_data = json.dumps(predict_dict, sort_keys=False)
    return Response(json_data, mimetype='application/json')
    # return jsonify(predict_dict)


# test path
@app.route('/test', methods=['post'])
def test():
    if not checkFile():
        return jsonify({'ERROR': 'File not uploaded!'})
    if not checkModel():
        return jsonify({'ERROR': 'Model not trained!'})

    file = request.files['file']
    sample_id_v, features_v, label_v = data_load(file)
    X, y = test_pre_process(features_v, label_v, res.selection_model)
    result = res.test(X, y)

    json_data = json.dumps(result, sort_keys=False)
    return Response(json_data, mimetype='application/json')
    # return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
