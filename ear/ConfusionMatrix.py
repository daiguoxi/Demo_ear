import os
import json
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import models
from models import resnet
from models import densenet
import torch.nn as nn


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN #round（，3取三位小数点）
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
    #绘图
    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 16,
                 }
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20,
                 }
        # print(self.num_classes,self.labels)
        plt.xticks(range(self.num_classes),self.labels, rotation=45,fontsize=6)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes),self.labels,fontsize=6)
        # 显示colorbar

        plt.colorbar()
        plt.xlabel('True Labels',font2)
        plt.ylabel('Predicted Labels',font2)
        plt.title('Confusion matrix')


        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        #plt.subplots_adjust(top=10, bottom=5, right=3, left=2, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig('confusionmatrix.pdf', format='pdf', dpi=600)
        plt.show()




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #模型训练的时候验证集的图像增强的一致

    data_transform = transforms.Compose([transforms.Resize(size=(448, 448)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #data_root = os.path.abspath(os.path.join(os.getcwd(), "C:/Users/lzz/Desktop/testSBS3/val"))  # get data root path
    #image_path = os.path.join(data_root, "SBS3G2", "SBS3G4")  # flower data set path
    #assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)
    image_path = "../nose Fig/"
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "nose Fig"),
                                            transform=data_transform)

    #img_dir='C:/Users/lzz/Desktop/testSBS3/val'
    #test_img=Image.open(img_dir)

    # 图像增强
    #validate_dataset = data_transform(test_img)
    batch_size = 4#原本为10
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)
    net = resnet.resnet50()
    # load pretrain weights
    model_weight_path = "./densenet121_best_model.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net=nn.DataParallel(net)
    #net.load_state_dict(torch.load(model_weight_path, map_location=device).module.state_dict())
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=7, labels=labels)
    net.eval()
    with torch.no_grad(): #使用这个函数停止梯度跟踪
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()
