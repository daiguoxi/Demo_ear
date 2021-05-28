import torch
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
import torch.nn as nn
#import earclassify
#import sys
#sys.path.insert(0, './earclassify')
#from earclassify.load_dataset import collate_fn, dataset,get_image_pd
#from earclassify.augmentation import *
#from earclassify.utils import PolyOptimizer,train,validate,creatdir
#from earclassify.models.vgg import vgg19 as Net
#import best_model


def predict(model,img_dir):
    #数据增强
    data_transforms = transforms.Compose([
            transforms.Resize(size=(400, 400)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ])
    #transform=data_transforms["val"]

    test_img=Image.open(img_dir)



    test_image_tensor = data_transforms(test_img)

    #test_image_tensor = test_image_tensor.view(1, 3, 224, 224)


    model.eval()
    out = model(test_image_tensor)

    print(out.data)
if __name__=="__main__":
    model=torch.load('resultsnew/densenet1215/best_model.pth')
    #model = best_model(pretrained=True)
    #exit()
    #model.load_state_dict(torch.load('best_model.pth'))
    predict(model,'/home/lzz/Ear/TMC/201912142391156.jpg')

