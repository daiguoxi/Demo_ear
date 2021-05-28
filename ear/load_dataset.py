# coding = utf-8
# @File    : load_dataset.py


import torch
import torch.utils.data as data
import cv2
import glob
import pandas as pd


# 字母标记名称(文件夹名字）
defect_label_order = ["CME","CSOM","EACB", "IC","NE","OE","SOM", "TMC" ]
# 与字母标记一一对应(类别名字）
defect_code = {
    "cholestestoma of middle ear": "CME",
    "chromic suppurative otits media":"CSOM",
    "external auditory cana bleeding":"EACB",
    "impacted cerumen":"IC",
    "normal eardrum":"NE",
    "otomycosis external":"OE",
    "secretory otitis media":"SOM",
    "tympanic membrane calcification":"TMC"
}
# 与数字标记一一对应(类别与数字标签一一对应）
defect_label = {
    "CME": "0",
    "CSOM": "1",
    "EACB": "2",
    "IC": "3",
    "NE": "4",
    "OE": "5",
    "SOM": "6",
    "TMC": "7"
}


# 用字典存储类别名字和数字标记
label2defect_map = dict(zip(defect_label.values(), defect_label.keys()))

# 获取图片路径
def get_image_pd(img_root):  # img-root = '/home/lzz/Ear' a
    # 利用glob指令获取图片列表（/*的个数根据文件构成确定）获取完整路径
    img_list = glob.glob(img_root + "/*/*.jpg")
    # print(img_list)
    # 利用DataFrame指令构建图片列表的字典，即图片列表的序号与其路径一一对应
    image_pd = pd.DataFrame(img_list, columns=["ImageName"])
    # 获取文件夹名称，也可以认为是标签名称
    image_pd["label_name"]=image_pd["ImageName"].apply(lambda x:x.split("/")[-2])
    # 将标签名称转化为数字标记
    image_pd["label"]=image_pd["label_name"].apply(lambda x:defect_label[x])
    print(image_pd["label"].value_counts())
    # print(image_pd)
    return image_pd

# 数据集
class dataset(data.Dataset):
    # 参数预定义
    def __init__(self, anno_pd, transforms=None,debug=False,test=False):
        # 图像路径
        self.paths = anno_pd['ImageName'].tolist()
        # 图像数字标签
        self.labels = anno_pd['label'].tolist()
        # 数字增强
        self.transforms = transforms
        # 程序调试
        self.debug=debug
        # 判定是否训练或测试
        self.test=test
    # 返回图片个数
    def __len__(self):
        return len(self.paths)
    # 获取每个图片
    def __getitem__(self, item):
        # 图像路径
        img_path =self.paths[item]
        # 读取
        img =cv2.imread(img_path)
        # 格式转换
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # 是否进行数据增强
        if self.transforms is not None:
            img = self.transforms(img)
        # 图像对应标签
        label = self.labels[item]
        # tensor和对应标签
        return torch.from_numpy(img).float(), int(label)

# 整理图片
def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label






