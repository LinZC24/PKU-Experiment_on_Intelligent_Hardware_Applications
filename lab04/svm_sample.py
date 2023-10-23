import matplotlib.pyplot as plt
from sklearn import datasets , svm, metrics

digits = datasets.load_digits() # 手 写 数 字 数 据 集 为 8x8 的 图 片， 16 级 灰 度
# digits.images 是 图 片 数 据， digits.target 是 标 签
images_and_labels = list(zip(digits.images , digits.target))
# 使 用 matplotlib 绘 制 前4个 样 本
for index , (image , label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image , cmap=plt.cm.gray_r , interpolation='nearest')
    plt.title('Training: %i' % label)

n_samples = len(digits.images)
# 将 二 维 数 据 变 成 一 维
data = digits.images.reshape((n_samples , -1))

# 建 立 分 类 器
classifier = svm.SVC(kernel = 'rbf', gamma=0.001)

# 用 前 一 半 数 据 进 行 训 练
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

expected = digits.target[n_samples // 2:]
# 预 测
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n" % (classifier , metrics.classification_report(expected , predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected , predicted))

# 使 用 matplotlib 绘 制 前4个 预 测 样 本
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index , (image , prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image , cmap=plt.cm.gray_r , interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()