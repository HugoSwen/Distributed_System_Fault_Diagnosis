from flask import Flask, request, jsonify

from static.Clf_object import *

app = Flask(__name__)


@app.route('/')
def hello():
    return 'hello'


# train path
@app.route('/train', methods=['post'])
def train():
    if 'file' not in request.files:
        return jsonify({'ERROR': 'File not uploaded!'})
    if request.files['file'].filename == '':
        return jsonify({'ERROR': 'File not uploaded!'})

    file = request.files['file']
    sample_id, features, label = data_load(file)
    X_train, y_train, selection_model = train_pre_process(features, label)

    res.train(X_train, y_train)
    res.is_trained = True
    res.selection_model = selection_model

    return 'success'


# test path
@app.route('/test', methods=['post'])
def test():
    if 'file' not in request.files:
        return jsonify({'ERROR': 'File not uploaded!'})
    if request.files['file'].filename == '':
        return jsonify({'ERROR': 'File not uploaded!'})
    if not res.is_trained:
        return jsonify({'ERROR': 'Model not trained!'})

    file = request.files['file']
    sample_id_v, features_v, label_v = data_load(file)
    X, y = test_pre_process(features_v, label_v, res.selection_model)
    report, matrix = res.test(X, y)

    result = {**report, **matrix}

    return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
