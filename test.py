import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
import glob
import cv2
import sys
import rasterio


def min_max_normalize(array):

    # 将两个数组进行最小-最大归一化
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    return normalized_array

def normalized(array):
    # 归一化操作
    normalized_array = 255 * (array - np.min(array)) / (np.max(array) - np.min(array))
    normalized_array = normalized_array.astype(np.uint8)
    return normalized_array

def mndwi_cal(img):
    """
    为图像添加坐标信息，
    """
    red = img[:,:,0]
    green = img[:,:,1]
    green_red = green + red
    green_red[green_red==0] = 1
    mndwi = (green - red) / green_red

    return mndwi

def resize_img(img, s=768):
    new_img = np.zeros_like(img)
    new_img = cv2.resize(new_img, (s,s), interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[0:2]
    if h>s or w>s:
        maxID = np.array([h,w]).argmax()
        if maxID==0:
            h1 = s
            w1 = int((w/h)*s)
        elif maxID==1:
            w1 = s
            h1 = int((h/w)*s)
        img = cv2.resize(img, (w1,h1), interpolation=cv2.INTER_NEAREST)
        h = h1
        w = w1
    h_up = (s-h)//2
    w_left = (s-w)//2
    d = len(img.shape)
    if d>2:
        new_img[h_up:h_up+h,w_left:w_left+w,:] = img
    elif d==2:
        new_img[h_up:h_up+h,w_left:w_left+w] = img
    return new_img, h_up, w_left, h, w

def rollback(new_img,h_up,w_left, new_h, new_w,old_h,old_w):
    d = len(new_img.shape)
    if d==3:
        old_img = new_img[h_up:h_up+new_h,w_left:w_left+new_w,:]
    elif d==2:
        old_img = new_img[h_up:h_up+new_h,w_left:w_left+new_w]

    old_img = cv2.resize(old_img, (old_w,old_h), interpolation=cv2.INTER_NEAREST)
    return old_img

def preprocess_image(image_path,s):
    transform = transforms.Compose([
        transforms.ToTensor()])
    with rasterio.open(image_path) as src:
        bands = [src.read(i) for i in range(1, 4)]  # 读取1、2、3波段
    # 将波段数据转换为 NumPy 数组
    red = bands[0]
    green = bands[1]
    blue = bands[2]
    tiff_RGB = np.stack([red, green, blue], axis=-1)
    tiff_RGB_normalize = normalized(tiff_RGB)
    old_h, old_w= red.shape
    img,h_up,w_left, new_h, new_w = resize_img(tiff_RGB_normalize, s)
    new_img = img
    tiff_RGB, _, _, _, _ = resize_img(tiff_RGB, s)
    # 计算mndwi
    mndwi = mndwi_cal(tiff_RGB)
    # 归一化
    mndwi = min_max_normalize(mndwi)*1.0  # 归一化，img数值归一化，故mndwi也要归一化        
    mndwi = mndwi[np.newaxis, :, : ]
    mndwi = torch.tensor(mndwi).to(torch.float32)           
    img = Image.fromarray(img)
    img = transform(img)
    img_mndwi = torch.cat((img, mndwi), dim=0).unsqueeze(0) # 合并img与mndwi
    return img_mndwi,h_up,w_left, new_h, new_w,old_h,old_w,new_img


# 测试模型
def test(img_path,model_path,device,s=512,show=True):
    # 加载训练好的模型
    from network import MigSwinUNet
    model = MigSwinUNet(img_size=512, patch_size=8, in_chans=4, out_chans=2,
             embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
             window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
             norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
             use_checkpoint=False, fused_window_process=False) 
    model.load_state_dict(torch.load(model_path,map_location=device,weights_only=True))
    model.to(device)
    # 加载测试图像
    test_image,h_up,w_left, new_h, new_w,old_h,old_w,new_img = preprocess_image(img_path,s)
    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        test_image = test_image.to(device)
        output = model(test_image)

    # 将输出转换为预测的分割结果
    predict = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() # 显示类别
    predict = predict.astype(np.uint8)
    predict = rollback(predict,h_up,w_left, new_h, new_w,old_h,old_w)
    return predict

    




