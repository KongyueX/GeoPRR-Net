"""
    写成输入两张图片输出一个数值的形式
"""

from dataloader import  Letterbox
from vitTranforms.meterCilp import meterClip
import torch
from PIL import Image
from torchvision import transforms

import os, sys
filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath)

class meterFormer():
    def __init__(self, weights, device):
        self.weights = weights
        self.device  = device

        self.model =    meterClip(
                        image_size = 256,
                        patch_size = 32,
                        num_classes = 512,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048,
                        channels=1,
                        dropout = 0.1,
                        emb_dropout = 0.1    

        ).to(device)
        
        self.model.load_state_dict(torch.load(weights, map_location=device))
        
    def _processImg(self, img1, img2):

        transform_binary  = transforms.Compose([
                            Letterbox(256),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5], std=[0.5])]
        )

        transform_rgb     = transforms.Compose([
                            Letterbox(256),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        
        img1 = transform_binary(img1) 
        img2 = transform_binary(img2)    

        return img1.unsqueeze(0).to(self.device) ,img2.unsqueeze(0).to(self.device)

    #-O 这一段代码要重写
    def Inference(self, img1, img2, label):
        img1, img2 = self._processImg(img1, img2)
        self.model.eval()
        return self.model(img1, img2, label)



if __name__ == "__main__":
    img1_path = ".\\data\\val\\image1\\0c3f8e6b795c4fc3.png"
    img2_path = ".\\data\\val\\image2\\0c3f8e6b795c4fc3.jpg"

    img1 = Image.open(img1_path).convert('L')  # 强制单通道（二值化图像）
    img2 = Image.open(img2_path).convert('L')  # 强制三通道（RGB）

    label_text = list(map(str, range(101)))

    weights = "result//best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds = meterFormer(weights, device)


    #-O 倒数第二层
    # target_layer = preds.model.decoder.transformer.layers[-1][1]
    
    #-O 初始化 Grad-CAM
    # grad_cam = GradCAM(preds.model, target_layer)
    label_g = preds.model.texteconder.tokenize(label_text).to(device)
    img1, img2 = preds.processImg(img1, img2)
    preImg ,preText = preds.Inference(img1, img2, label_g)

    #-O 效果很nice
    imgIndex = preImg.argmax(dim=-1).detach().cpu().numpy().squeeze()
    endNum = label_text[imgIndex]
    print(endNum)
    # #-O 生成热力图
    # heatmap, class_idx = grad_cam(img1, img2)

    # #-O 这里的热力图显示的效果不对，初步推断为在模型前向传播的过程中，图像的空间结果已经被破坏
    # superimposed_img = grad_cam.visualize_gradcam(".\\data\\val\\image2\\0c3f8e6b795c4fc3.jpg", heatmap, class_idx)

