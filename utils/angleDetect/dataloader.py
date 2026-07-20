import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm 
import torch.nn.functional as F

import os, sys
filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath)

class Letterbox:
    def __init__(self, size=224, fill=0):
        self.size = size
        self.fill = fill  

    def __call__(self, img):
        mode = img.mode  # 'RGB' 或 'L'

        w, h = img.size
        ratio = min(self.size / w, self.size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        new_img = Image.new(mode, (self.size, self.size), self.fill)
        new_img.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        
        return new_img
    
"""
    自定义语义分割数据加载器
"""    
class CustomSegDataset(Dataset):
    def __init__(self, img1_list, img2_list):
        self.img1_list = img1_list
        self.img2_list = img2_list

        self.transformGary = transforms.Compose([
                             Letterbox(256),    
                             #-O 这里deepseek说了会自动归一化到0,1     
                             transforms.ToTensor(),
                             #-O 这里直接就是标签不需要Normalize
                             transforms.Lambda(lambda x: (x > 0.5).float())]
        
        )

        self.transformRgb = transforms.Compose([
                            Letterbox(256),         
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        
        )

    def __len__(self):
        return len(self.img1_list)

    def __getitem__(self, idx):
        img1 = self.img1_list[idx]
        img2 = self.img2_list[idx]

        img1 = Image.open(img1).convert('L')
        img2 = Image.open(img2).convert('RGB')

        img1 = self.transformGary(img1)
        img2 = self.transformRgb(img2)

        return img1, img2


"""
    自定义表计识别数据加载器
"""
#-O 这段代码默认是把数据都转换为RGB类型，后续需要修改（可以添加按钮，数据集足够时使用RGB,数据集不充分时，使用灰度图）
class CustomDataset(Dataset):
    def __init__(self, img1_list, img2_list, lbl_list, transform="L"):
        self.img1_list = img1_list
        self.img2_list = img2_list
        self.lbl_list  = lbl_list
        self.switch = transform
        if self.switch == "L":
            self.transform =transforms.Compose([
                            Letterbox(256),         
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5], std=[0.5])]
            
            )
        else:
            self.transform =transforms.Compose([
                            Letterbox(256),         
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            
            )

    def __len__(self):
        return len(self.img1_list)

    def __getitem__(self, idx):
        img1 = self.img1_list[idx]
        img2 = self.img2_list[idx]
        label = self.lbl_list[idx]

        if self.switch == "L":
            img1 = Image.open(img1).convert('L')
            img2 = Image.open(img2).convert('L')
        #-O 理论上img1是二值图不能这么转化的。
        else:
            img1 = Image.open(img1).convert('RGB')
            img2 = Image.open(img2).convert('RGB')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        with open(label, 'r') as file:
            label = int(file.read().strip())
        # label = label 

        return img1, img2, label
    
    
class dataset_list():
    def __init__(self):
        self.path = os.getcwd()
    

    def load_dataset_folder(self, img1, img2, label=None):
        valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        data1, data2, labels = [], [], []

        img1_dir = os.path.join(self.path, img1)
        data1 = [os.path.join(img1_dir, f)
                for f in os.listdir(img1_dir)
                if f.endswith(valid_image_extensions)]
        
        img2_dir = os.path.join(self.path, img2)
        data2 = [os.path.join(img2_dir, f)
                for f in os.listdir(img2_dir)
                if f.endswith(valid_image_extensions)]

        if label and label.strip():  
            labels_dir = os.path.join(self.path, label)
            labels = [os.path.join(labels_dir, f)
                    for f in os.listdir(labels_dir)
                    if f.endswith('.txt')]
        else:
            labels = [] 

        return list(data1), list(data2), list(labels)

if __name__ == "__main__":
    datalist = dataset_list()
    img1_list, img2_list, label_list =  datalist.load_dataset_folder(  # 调用方法，而不是对象本身
                                        img1="data\\train\\image1",
                                        img2="data\\train\\image2",
                                        label="data\\train\\labels"
    
    )

    datasets = CustomDataset(img1_list= img1_list, img2_list= img2_list, lbl_list= label_list)
    dataloader = DataLoader(datasets, batch_size=16, shuffle=True)

    for epoch in range(10):
        # 添加total参数显示总批次数，设置进度条宽度和样式
        for batch_idx, (img1, img2, label) in tqdm(enumerate(dataloader),
                                                total=len(dataloader),
                                                desc=f"Epoch {epoch+1}/10",
                                                ncols=100,
                                                ascii='->='):
            # 训练代码放在这里
            pass