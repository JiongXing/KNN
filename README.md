# K-近邻

K-近邻算法，(kNN，k-NearestNeighbor)分类算法是数据挖掘分类技术中最简单的方法之一。
所谓K最近邻，就是k个最近的邻居的意思，说的是每个样本都可以用它最接近的k个邻居来代表。

scikit-learn内置了一些数据集，其中有个比较经典的鸢尾花数据集**(iris)**，我们可以用它来体验一下KNN分类。
这个数据集有150条鸢尾花数据，鸢尾花一共有三种分类，我们的任务是根据鸢尾花数据的特征，判别出它是哪种类型的鸢尾花。

# 加载iris数据集
```
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data[:5])
```

输出前5条数据：
```
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
```

# 预处理：标准化
KNN会对所有特征计算做加和，而不同特征的值空间差异可能很大，如果直接相加，其结果将会被大值特征主导，而小值特征却影响甚小，无法体现特征的平等。因此KNN一般都需要进行标准化。
```
from sklearn.preprocessing import StandardScaler

# 对数据集进行标准化
std = StandardScaler()
data = std.fit_transform(iris.data)
separator.line("标准化后的数据")
print(data[:5])
```

输出标准化后的数据：
```
[[-0.90068117  1.03205722 -1.3412724  -1.31297673]
 [-1.14301691 -0.1249576  -1.3412724  -1.31297673]
 [-1.38535265  0.33784833 -1.39813811 -1.31297673]
 [-1.50652052  0.10644536 -1.2844067  -1.31297673]
 [-1.02184904  1.26346019 -1.3412724  -1.31297673]]
```

# 划分训练集与数据集
> KNN其实是不需要训练的，但是为了统一术语，依然这么称呼划分的两个集。

把data和target拆成两个集，75%作为训练集，25%作为测试集：
```
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test \
    = train_test_split(data, iris.target, test_size=0.25)
```

# KNN分类
scikit-learn实现了KNN算法，默认K值为5，我们这里就使用默认值，意即取最邻近的5个样本的target作为分类结果。

```
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(data_train, target_train)
```

# 评估准确率

KNN分类器也提供了准确率评估API，输入测试集数据与测试集目标，可以评估当前训练后(调用fit方法后)的knn：

```
score = knn.score(data_test, target_test)
print(score)
```

输出准确率。其结果每次运行都可能不一样，因为划分训练集与测试集每次都不一样：
```
0.9473684210526315
```

# 预测
调用predict方法，输出测试集数据，得出预测结果：

```
target_predict = knn.predict(data_test)
print(target_test)
print(target_predict)
```

输出：
```
****************************** 真实值 ******************************
[2 2 2 0 2 1 0 2 0 1 0 0 1 1 0 2 1 2 2 0 2 2 2 2 1 1 1 2 0 0 2 1 0 1 2 0 1 1]
****************************** 预测值 ******************************
[2 2 2 0 2 1 0 1 0 1 0 0 1 1 0 2 1 2 2 0 2 2 2 2 1 1 1 2 0 0 2 1 0 1 2 0 1 2]
```

这里，预测值与真实值大致相同。