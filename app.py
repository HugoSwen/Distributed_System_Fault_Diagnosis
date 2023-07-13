import os
import pickle
import tempfile
import threading
import time

import joblib
import pymysql
from flask import Flask, request, Response, session, send_file, make_response
from werkzeug.utils import secure_filename

from static.DLRFC import *
from static.config import database_config

app = Flask(__name__)
app.secret_key = 'Larghetto_se'
dlrfc = DLRFC()


def getcurcor():
    cnn = pymysql.connect(
        **database_config,
        database="fault_diagnosis_system"
    )
    cursor = cnn.cursor()
    return cursor, cnn


def checkfile():
    file = request.files['file']

    if file:
        filename = secure_filename(file.filename)
        extension = os.path.splitext(filename)[1]
        if extension not in ['.csv', '.pkl', '.joblib', 'pickle']:
            return False
    else:
        return False
    return True


# login path
@app.route("/login", methods=["post"])
def login():
    data = request.get_json()
    username = data["username"]
    password = data["password"]

    cursor, cnn = getcurcor()
    sql = "select * from user where username= %s and password= %s"
    cursor.execute(sql, (username, password))
    result = cursor.fetchone()
    cursor.close()
    cnn.close()

    if result:
        session['username'] = username
        return "success"
    else:
        return "failure"


# register path
@app.route("/register", methods=["post"])
def register():
    data = request.get_json()
    username = data["username"]
    password = data["password"]

    cursor, cnn = getcurcor()
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


# train path
@app.route('/train', methods=['post'])
def train():
    if not checkfile():
        return "failure"

    file = request.files['file']
    dlrfc.train(file)

    # cursor, cnn = getcurcor()
    # username = session.get("username")
    # sql = "update user set model_blob = %s where username = %s"
    # try:
    #     cursor.execute(sql, (pickle.dumps(dlrfc), username))
    #     cnn.commit()
    # except pymysql.Error as e:
    #     cnn.rollback()
    #     print(e)
    # cursor.close()
    # cnn.close()

    return 'success'


# validate path
@app.route('/validate', methods=['post'])
def validate():
    if not checkfile():
        return "failure"

    file = request.files['file']
    result = dlrfc.validate(file)

    json_data = json.dumps(result, sort_keys=False)
    return Response(json_data, mimetype='application/json')


# test path
@app.route('/test', methods=['post'])
def test():
    if not checkfile():
        return "failure"

    # cursor, cnn = getcurcor()
    # username = session.get("username")
    # sql = "select * from user where username = %s"
    # cursor.execute(sql, username)
    # row = cursor.fetchone()
    # cursor.close()
    # cnn.close()
    #
    # if row[3]:
    #     dlrfc = pickle.loads(row[3])
    # else:
    #     return "failure"

    file = request.files['file']
    result = dlrfc.test(file)

    json_data = json.dumps(result, sort_keys=False)
    return Response(json_data, mimetype='application/json')


# download path
@app.route('/download', methods=['post'])
def download():
    # cursor, cnn = getcurcor()
    # username = session.get("username")
    # sql = "select * from user where username = %s"
    # cursor.execute(sql, username)
    # row = cursor.fetchone()

    if dlrfc.selection_model:
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False, dir="tempfiles") as temp_file:
            file_path = temp_file.name
            joblib.dump(dlrfc, file_path)
        #     print(file_path)
        # response = make_response(send_file(file_path, as_attachment=True))
        response = Response(pickle.dumps(dlrfc), mimetype='application/octet-stream')

        response.headers.set('Content-Disposition', 'attachment', filename='model.joblib')

        # def delete_file(file_path):
        #     time.sleep(1)
        #     os.remove(file_path)
        #
        # # 创建一个新的线程来执行文件删除操作
        # delete_thread = threading.Thread(target=delete_file, args=(file_path,))
        # delete_thread.start()

        return response
    else:
        return "failure"


# upload path
@app.route('/upload', methods=['post'])
def upload():
    if not checkfile():
        return "failure"

    file = request.files['file']
    global dlrfc
    dlrfc = joblib.load(file)
    return "success"


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
