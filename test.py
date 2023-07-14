import joblib

dlrfc = joblib.load("modelfiles/wangyule.joblib")# 填入下载的模型路径

dlrfc.train("../data_new/train_10000.csv")# 训练数据路径

dlrfc.validate("../data_new/validate_1000.csv")# 验证数据路径

result = dlrfc.test("../data_new/test_2000_x.csv")# 测试数据路径
