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

"""
test.py

Model testing file.
This script is used to evaluate the trained model.
"""

def min_max_normalize(array):
    # Apply min-max normalization to the input array
    normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
    return normalized_array


def normalized(array):
    # Normalize the array to the range [0, 255]
    normalized_array = 255 * (array - np.min(array)) / (np.max(array) - np.min(array))
    normalized_array = normalized_array.astype(np.uint8)
    return normalized_array


def mndwi_cal(img):
    """
    Calculate the MNDWI (Modified Normalized Difference Water Index)
    from the input image.
    """
    red = img[:, :, 0]
    green = img[:, :, 1]
    green_red = green + red
    green_red[green_red == 0] = 1
    mndwi = (green - red) / green_red

    return mndwi


def resize_img(img, s=768):
    # Create a blank image with the target size
    new_img = np.zeros_like(img)
    new_img = cv2.resize(new_img, (s, s), interpolation=cv2.INTER_NEAREST)

    h, w = img.shape[0:2]

    # Resize the image while preserving the aspect ratio if necessary
    if h > s or w > s:
        maxID = np.array([h, w]).argmax()
        if maxID == 0:
            h1 = s
            w1 = int((w / h) * s)
        elif maxID == 1:
            w1 = s
            h1 = int((h / w) * s)

        img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_NEAREST)
        h = h1
        w = w1

    # Compute the top and left padding offsets
    h_up = (s - h) // 2
    w_left = (s - w) // 2

    # Paste the resized image into the center of the new image
    d = len(img.shape)
    if d > 2:
        new_img[h_up:h_up + h, w_left:w_left + w, :] = img
    elif d == 2:
        new_img[h_up:h_up + h, w_left:w_left + w] = img

    return new_img, h_up, w_left, h, w


def rollback(new_img, h_up, w_left, new_h, new_w, old_h, old_w):
    # Crop the valid region from the padded image
    d = len(new_img.shape)
    if d == 3:
        old_img = new_img[h_up:h_up + new_h, w_left:w_left + new_w, :]
    elif d == 2:
        old_img = new_img[h_up:h_up + new_h, w_left:w_left + new_w]

    # Resize the cropped image back to the original size
    old_img = cv2.resize(old_img, (old_w, old_h), interpolation=cv2.INTER_NEAREST)
    return old_img


def preprocess_image(image_path, s):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    with rasterio.open(image_path) as src:
        bands = [src.read(i) for i in range(1, 4)]  # Read bands 1, 2, and 3

    # Convert the band data to a NumPy array
    red = bands[0]
    green = bands[1]
    blue = bands[2]
    tiff_RGB = np.stack([red, green, blue], axis=-1)

    # Normalize the RGB image to [0, 255]
    tiff_RGB_normalize = normalized(tiff_RGB)

    old_h, old_w = red.shape
    img, h_up, w_left, new_h, new_w = resize_img(tiff_RGB_normalize, s)
    new_img = img

    # Resize the original RGB image for MNDWI calculation
    tiff_RGB, _, _, _, _ = resize_img(tiff_RGB, s)

    # Calculate MNDWI
    mndwi = mndwi_cal(tiff_RGB)

    # Normalize MNDWI to match the image preprocessing scale
    mndwi = min_max_normalize(mndwi) * 1.0
    mndwi = mndwi[np.newaxis, :, :]
    mndwi = torch.tensor(mndwi).to(torch.float32)

    # Convert the image to tensor format
    img = Image.fromarray(img)
    img = transform(img)

    # Concatenate the RGB image and MNDWI as model input
    img_mndwi = torch.cat((img, mndwi), dim=0).unsqueeze(0)

    return img_mndwi, h_up, w_left, new_h, new_w, old_h, old_w, new_img


# Test the model
def test(img_path, model_path, device, s=512, show=True):
    # Load the trained model
    from network import MigSwinUNet
    model = MigSwinUNet(
        img_size=512, patch_size=8, in_chans=4, out_chans=2,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
        use_checkpoint=False, fused_window_process=False
    )

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    # Load and preprocess the test image
    test_image, h_up, w_left, new_h, new_w, old_h, old_w, new_img = preprocess_image(img_path, s)

    # Perform prediction
    model.eval()
    with torch.no_grad():
        test_image = test_image.to(device)
        output = model(test_image)

    # Convert the output to the predicted segmentation map
    predict = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Predicted class map
    predict = predict.astype(np.uint8)
    predict = rollback(predict, h_up, w_left, new_h, new_w, old_h, old_w)

    return predict 




