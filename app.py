import json
import os

import pymysql
from flask import Flask, request, jsonify, Response
from werkzeug.utils import secure_filename

from static.Clf_object import *

app = Flask(__name__)


def getCurcor():
    data = request.get_json()
    username = data["username"]
    password = data["password"]
    connection = pymysql.connect(host="localhost",
                                 port=3306,
                                 user="root",
                                 password="yangqiyuan2003",
                                 charset="utf8mb4",
                                 database="demo1"
                                 )
    cursor = connection.cursor()
    return username, password, cursor, connection


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
@app.route("/do_register", methods=["post"])
def do_register():
    username, password, cursor, connection = getCurcor()
    sql = "insert into user values(%s,%s)"
    try:
        cursor.execute(sql, (username, password))
        connection.commit()
        cursor.close()
        connection.close()
        return "success"
    except:
        errorDefault = ""
        connection.rollback()
        sql_2 = "select * from user where username=%s"
        cursor.execute(sql_2, username)
        result = cursor.fetchone()
        if (result):
            errorDefault = "your username has been registered"
        else:
            errorDefault = "false"
        cursor.close()
        connection.close()
        return errorDefault


# login path
@app.route("/do_login", methods=["post"])
def do_login():
    username, password, cursor, connection = getCurcor()
    sql = "select * from user where username=%s and password=%s"
    cursor.execute(sql, (username, password))
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    if result:
        return username + ":  success"
    else:
        return "false"


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
