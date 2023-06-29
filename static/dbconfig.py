import pymysql

database_config = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "charset": "utf8",
    "port": 3306,
}
try:
    # 建立数据库连接
    cnn = pymysql.connect(**database_config, **{"database" : "db02"})

    # 连接成功
    print("Database connection successful!")

    # 执行其他数据库操作...
    sql = "select * from emp"
    cursor = cnn.cursor()

    # 关闭数据库连接
    cnn.close()

except pymysql.Error as e:
    # 连接失败，处理异常
    print("Database connection failed:", str(e))


