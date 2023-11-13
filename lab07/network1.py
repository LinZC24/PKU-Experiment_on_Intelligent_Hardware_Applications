# 此代码用于训练及测试神经网络
# 需要在此代码所在的目录下存放一个数据集文件夹
# 数据集文件夹包含“train”、“test”两个子文件夹
# 两个子文件夹下均有四个以手势类型命名的文件夹，存放着样本
# 训练部分使用小批量梯度下降、Adam优化器
# 使用tensorboard的SummaryWriter记录每轮训练后在训练集、测试集上的损失、准确率等情况
# SummaryWriter、训练好的模型将保存在数据集文件夹下
# 测试部分将给出模型在测试集上的混淆矩阵
# 混淆矩阵图片将保存在数据集文件夹下

DATA_FOLDER = 'dataset'  # 数据集文件夹名
DO_TRAIN = True  # 是否进行训练，True需要有数据集
DO_TEST = True  # 是否进行测试，True需要DO_TRAIN为True或数据集文件夹下已存在一个网络模型
SAVE_MODEL = True  # 是否保存训练好的模型，True需要DO_TRAIN为True
SAVE_CMFIG = True  # 是否保存混淆矩阵，True需要DO_TEST为True
LEARNING_RATE = 0.001  # Adam优化器学习率
MAX_EPOCH = 10  # 进行epoch数
BATCH_SIZE = 20  # 每批含样本数

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import itertools
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

writer = SummaryWriter()  # 用于记录训练
torch.set_grad_enabled(True)

train_set = torchvision.datasets.ImageFolder(f'./{DATA_FOLDER}/train', transform=transforms.Compose(
    [transforms.Grayscale(1), transforms.ToTensor()]))  # 从数据集文件夹导入训练样本，灰度化并转为张量
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True)  # 训练集加载器，随机打乱训练集并以BATCH_SIZE个为一批


# 神经网络
# 采用“输入-卷积-池化-卷积-池化-一维化-全连接-输出”的结构
# 激活函数为ReLu
# 图片原始大小为80*60
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=5, stride=2)
        #self.fc1 = nn.Linear(in_features=12 * 3 * 2, out_features=60)
        #self.fc2 = nn.Linear(in_features=60, out_features=30)
        self.fc2 = nn.Linear(in_features=18 * 4 * 2, out_features=30)
        self.out = nn.Linear(in_features=30, out_features=4)
        #self.out = nn.Linear(in_features=60, out_features=4)

    def forward(self, t):
        t = F.relu(self.conv1(t))  # -> 38*28   76*56
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #t = F.max_pool2d(t, kernel_size=4, stride=4)  # ->19*14
        t = F.relu(self.conv2(t))  # -> 8*5
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # ->4*2
        t = t.reshape(-1, 18 * 4 * 2)
        #t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        return self.out(t)


# 训练部分
def train():
    network = Network()
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
    images, labels = next(iter(train_loader))  # 从前面的加载器中按批获得样本

    # 在tensorboard中记录一批样本的图像，添加网络，以监视样本或网络可能的异常
    grid = torchvision.utils.make_grid(images)  # 将一批样本做成网格
    tb = SummaryWriter(f'./{DATA_FOLDER}')  # SummaryWriter将保存在数据集文件夹下
    tb.add_image('images', grid)  # 添加样本网格
    #tb.add_graph(network, images)  # 添加网络  

    # 比较网络对一批样本的预测preds与样本的真实标签labels，返回预测正确的个数
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    # 训练MAX_EPOCH轮
    for epoch in range(MAX_EPOCH):
        t0 = time.time()  # 记录用时
        total_loss = 0  # 记录训练集上的损失
        total_correct = 0  # 记录训练集上的正确个数
        for batch in train_loader:  # 按批进行
            images, labels = batch
            preds = network(images)  # 获得目前网络对这一批的预测
            loss = F.cross_entropy(preds, labels)  # 计算交叉熵
            optimizer.zero_grad()  # 梯度清零，避免循环时backward()累加梯度
            loss.backward()  # 反向传播求解梯度
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 更新训练集上的损失
            total_correct += get_num_correct(preds, labels)  # 更新训练集上正确个数
        # 一次epoch完成，开始在测试集上看效果
        pred_set = torchvision.datasets.ImageFolder(f'./{DATA_FOLDER}/test', transform=transforms.Compose(
            [transforms.Grayscale(1), transforms.ToTensor()]))  # 按和前面train_set一样的方法获得测试集pred_set
        prediction_loader = torch.utils.data.DataLoader(pred_set, batch_size=BATCH_SIZE)  # 以及对应加载器
        p_total_loss = 0  # 记录在训练集上的损失
        p_total_correct = 0  # 记录训练集上总正确数
        for batch in prediction_loader:
            images, labels = batch
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            p_total_loss += loss.item()
            p_total_correct += get_num_correct(preds, labels)

        # 在终端显示epoch数、准确率、用时，监视训练进程
        print('epoch', epoch + 1, 'total_correct:',
              total_correct, 'loss:', total_loss)
        print('train_set accuracy:', total_correct / len(train_set))
        print('test_set accuracy:', p_total_correct / len(pred_set))
        print('time spent:', time.time() - t0)

        # 在SummaryWriter中分别记录训练集和测试集的损失、准确率、正确数信息
        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Prediction Loss', p_total_loss, epoch)
        tb.add_scalar('Number Correct', total_correct, epoch)
        tb.add_scalar('Prediction Number Correct', p_total_correct, epoch)
        tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)
        tb.add_scalar('Prediction Accuracy', p_total_correct / len(pred_set), epoch)
    tb.close()
    if SAVE_MODEL:  # 保存模型在数据集文件夹下
        torch.save(network, F'./{DATA_FOLDER}/network.pkl')


if DO_TRAIN:
    train()


# 测试模型，绘制混淆矩阵
def test(network):
    def get_all_preds(model, loader):
        # 传入网络模型和加载器，获得模型对加载器中样本的全部预测
        all_preds = torch.tensor([])
        for batch in loader:
            images, labels = batch
            preds = model(images)  # 按批获得预测
            all_preds = torch.cat((all_preds, preds), dim=0)  # 再拼接起来
        return all_preds

    with torch.no_grad():  # 测试时不需要计算梯度等数据，节省资源
        pred_set = torchvision.datasets.ImageFolder(f'./{DATA_FOLDER}/test', transform=transforms.Compose(
            [transforms.Grayscale(1), transforms.ToTensor()]))  # 导入测试集
        prediction_loader = torch.utils.data.DataLoader(pred_set, batch_size=BATCH_SIZE)  # 对应加载器，由于是测试，无需shuffle
        train_preds = get_all_preds(network, prediction_loader)  # 获取网络对测试集数据的预测
    total_types = len(pred_set.classes)  # 总分类数
    stacked = torch.stack((torch.tensor(pred_set.targets), train_preds.argmax(dim=1)), dim=1)  # 拼接测试样本的真实类型与预测类型
    cmt = torch.zeros(total_types, total_types, dtype=torch.int32)  # 初始化混淆矩阵
    for p in stacked:  # 根据测试样本的真实类型与预测类型，在混淆矩阵的对应位置计数+1
        train_label, predicted_label = p.tolist()
        cmt[train_label, predicted_label] = cmt[train_label, predicted_label] + 1

    def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
        # 传入ndarray型的混淆矩阵，分类名，标题，配色；绘制上文得到的混淆矩阵
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()  # 设置颜色渐变条
        tick_marks = np.arange(len(classes))  # 图像有分类数个刻度
        plt.xticks(tick_marks, classes, rotation=45)  # 用分类名标签xy刻度
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        # 在表格上对应位置显示数字，为可视化效果，以其中最大数据的一半为界确定文字颜色的黑白
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()  # 填充图像区域
        plt.ylabel('True label')  # 命名坐标
        plt.xlabel('Predicted label')

    # 绘图
    plt.figure(figsize=(4, 4))
    plot_confusion_matrix(cmt.numpy(), pred_set.classes)
    if SAVE_CMFIG:  # 将混淆矩阵图片保存在数据集文件夹下
        plt.savefig(f'./{DATA_FOLDER}/confusion_matrix.png')
    plt.show()


if DO_TEST:
    test(torch.load(f'./{DATA_FOLDER}/network.pkl'))
