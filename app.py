import json
import os

import pymysql
from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename

from static.Clf_object import *
from static.config import database_config

app = Flask(__name__)


def getcurcor():
    data = request.get_json()
    username = data["username"]
    password = data["password"]

    cnn = pymysql.connect(
        **database_config,
        database="fault_diagnosis_system"
    )
    cursor = cnn.cursor()
    return username, password, cursor, cnn


def checkfile():
    file = request.files['file']

    if file:
        filename = secure_filename(file.filename)
        extension = os.path.splitext(filename)[1]
        if extension != '.csv':
            return False
    else:
        return False
    return True


def checkmodel():
    if res.selection_model is None:
        return False
    return True


# register path
@app.route("/register", methods=["post"])
def register():
    username, password, cursor, cnn = getcurcor()
    sql = "insert into user(username,password) values(%s,%s)"
    try:
        cursor.execute(sql, (username, password))
        cnn.commit()
        result = "success"
    except pymysql.Error as e:
        cnn.rollback()
        print(e)
        result = "failure"

    cursor.close()
    cnn.close()
    return result


# login path
@app.route("/login", methods=["post"])
def login():
    username, password, cursor, cnn = getcurcor()
    sql = "select * from user where username= %s and password= %s"
    cursor.execute(sql, (username, password))
    result = cursor.fetchone()
    cursor.close()
    cnn.close()
    if result:
        return username + ":  success"
    else:
        return "failure"


# modify path
@app.route("/update", methods=["post"])
def update():
    username, password, cursor, cnn = getcurcor()
    sql1 = "update user set password = %s where username = %s"
    sql2 = "select * from user where username = %s"
    cursor.execute(sql2, username)
    if cursor.fetchone():
        cursor.execute(sql1, (password, username))
        cnn.commit()
        result = "success"
    else:
        result = "failure"
    cursor.close()
    cnn.close()
    return result


# train path
@app.route('/train', methods=['post'])
def train():
    if not checkfile():
        return "failure"

    file = request.files['file']
    sample_id, features, label = data_load(file)
    X_train, y_train, selection_model = train_pre_process(features, label)

    res.train(X_train, y_train)
    res.selection_model = selection_model

    return 'success'


# test path
@app.route('/test', methods=['post'])
def test():
    if not checkfile() or not checkmodel():
        return "failure"

    file = request.files['file']
    sample_id_v, features_v, label_v = data_load(file)
    X, y = test_pre_process(features_v, label_v, res.selection_model)
    result = res.test(X, y)

    json_data = json.dumps(result, sort_keys=False)
    return Response(json_data, mimetype='application/json')
    # return jsonify(result)


# predict path
@app.route('/predict', methods=['post'])
def predict():
    if not checkfile() or not checkmodel():
        return "failure"

    file = request.files['file']
    sample_id_v, features_v, label_v = data_load(file)
    X, y = test_pre_process(features_v, label_v, res.selection_model)
    result = res.predict(X)

    json_data = json.dumps(result, sort_keys=False)
    return Response(json_data, mimetype='application/json')
    # return jsonify(predict_dict)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
