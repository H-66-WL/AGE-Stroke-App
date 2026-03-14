import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

# 读取新的训练数据
train_data = pd.read_csv("training_data_new.csv")  # 请确认文件名

# 定义特征列
feature_cols = ['FOS', 'PTGS2', 'LMNB1', 'CXCL1']
X_train = train_data[feature_cols]
y_train = train_data['GroupType']

# 检查标签
print("标签类别：", y_train.unique())

# 训练模型
model = GaussianNB()
model.fit(X_train, y_train)

# 保存模型（覆盖原来的 nb_model.pkl）
joblib.dump(model, 'nb_model.pkl')
print("✅ 新模型已保存为 nb_model.pkl")