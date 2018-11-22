from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def print_line(title):
    print("*" * 30 + " {} ".format(title) + "*" * 30)


iris = load_iris()
print_line("前5条数据")
print(iris.data[:5])

# 对数据集进行标准化
std = StandardScaler()
data = std.fit_transform(iris.data)
print(data[:5])

# 把data和target拆成两个集，75%作为训练集，25%作为测试集
data_train, data_test, target_train, target_test \
    = train_test_split(data, iris.target, test_size=0.25)

# KNN分类器
knn = KNeighborsClassifier()
knn.fit(data_train, target_train)

# 评估准确率
print_line("准确率")
score = knn.score(data_test, target_test)
print(score)

# 预测
target_predict = knn.predict(data_test)
print_line("真实值")
print(target_test)
print_line("预测值")
print(target_predict)
