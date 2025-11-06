import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, random_split
import glob
import os
import numpy as np
import cv2
from PIL import Image
from settings import *
import matplotlib.pyplot as plt
import rasterio
'''
Data loading
'''

def min_max_normalize(array):
	normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))
	return normalized_array

def normalized(array):
    normalized_array = 255 * (array - np.min(array)) / (np.max(array) - np.min(array))
    normalized_array = normalized_array.astype(np.uint8)
    return normalized_array

def mndwi_cal(img):
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
	return new_img


def random_horizontal_flip(image, mndwi, mask):
	seed = torch.rand(1)
	# print(seed)
	if seed < 0.3:
		image = TF.hflip(image)
		mndwi = TF.hflip(mndwi)
		mask = TF.hflip(mask)
	elif seed > 0.7:
		image = TF.vflip(image)
		mndwi = TF.vflip(mndwi)
		mask = TF.vflip(mask)
	return image, mndwi, mask


opt = net_config
class TUDataset(Dataset):
	def __init__(self,imgFile=opt["img_path"], labelFile=opt['label_path'],transforms_img=None,isTrain=True):
		self.isTrain = isTrain

		if transforms_img == None:
			self.imgTransform = transforms.Compose([
				transforms.ToTensor(),
				# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
				])
		else:
			self.imgTransform = transforms_img
		imgs = glob.glob(os.path.join(imgFile,'*'+opt['end_with'][0]))
		self.imgs = sorted(imgs)
		labels = glob.glob(os.path.join(labelFile,'*'+opt['end_with'][0]))
		self.labels =  sorted(labels)
		self.len = len(self.imgs)


	def __getitem__(self,index):
		idx = index % self.len
		if self.isTrain:
			with rasterio.open(self.imgs[idx]) as src:
				bands = [src.read(i) for i in range(1, 4)] 
			red = bands[0]
			green = bands[1]
			blue = bands[2]
			tiff_RGB = np.stack([red, green, blue], axis=-1)
			tiff_RGB_normalize = normalized(tiff_RGB)
			img = resize_img(tiff_RGB_normalize, s=512)
			tiff_RGB = resize_img(tiff_RGB, s=512)
			mndwi = mndwi_cal(tiff_RGB)
			mndwi = min_max_normalize(mndwi)*1.0	
			mndwi = mndwi[np.newaxis, :, : ]
			mndwi = torch.tensor(mndwi).to(torch.float32)			
			img = Image.fromarray(img)
			img = self.imgTransform(img)
			with rasterio.open(self.labels[idx]) as src:
				mask = src.read(1)
			mask = resize_img(mask, s=opt['img_height'])
			mask = torch.from_numpy(mask).long() 
			img, mndwi, mask = random_horizontal_flip(img, mndwi, mask) 
			img_mndwi = torch.cat((img, mndwi), dim=0) 
			return {'img':img_mndwi,'mask':mask}
			# return {'img':img_mndwi}
		else:
			with rasterio.open(self.imgs[idx]) as src:
				bands = [src.read(i) for i in range(1, 4)]
			red = bands[0]
			green = bands[1]
			blue = bands[2]
			tiff_RGB = np.stack([red, green, blue], axis=-1)
			tiff_RGB_normalize = normalized(tiff_RGB)
			img = resize_img(tiff_RGB_normalize, s=512)
			tiff_RGB = resize_img(tiff_RGB, s=512)
			red_resize = tiff_RGB[:,:,0]
			green_resize = tiff_RGB[:,:,1]
			blue_resize = tiff_RGB[:,:,2]
			a = green_resize+red_resize
			a[a==0] = 1
			mndwi = (green_resize-red_resize)/a
			mndwi = min_max_normalize(mndwi)*1.0  	
			mndwi = mndwi[np.newaxis, :, : ]
			mndwi = torch.tensor(mndwi).to(torch.float32)			
			img = Image.fromarray(img)
			img = self.imgTransform(img)
			img_mndwi = torch.cat((img, mndwi), dim=0) 
			return {'img':img_mndwi}
	def __len__(self):
		return len(self.imgs)

if __name__ == "__main__":
	imgFile=opt["img_path"]
	labelFile = opt['label_path']
	transforms_img = transforms.Compose([
		transforms.ToTensor()
		])
	full_dataset = TUDataset(imgFile,labelFile,transforms_img=transforms_img,isTrain=True)  
	train_ratio = 1.0
	test_ratio = 1 - train_ratio
	train_size = int(train_ratio * len(full_dataset))
	test_size = len(full_dataset) - train_size
	train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
	trainLoader = DataLoader(
		train_dataset,
		batch_size=1,
		shuffle=True,
		num_workers=opt['worker_num']
		)
	iterNG = trainLoader.__iter__()
	lenNum = len(trainLoader)
	for i in range(0,lenNum):
		batchData = iterNG.__next__()
		img = batchData['img'].to(opt['device'])
		mask = batchData['mask'].to(opt['device'])
		mask = mask.squeeze(0)
		a = img[:,-1,:,:]
		a = torch.squeeze(a)
		b = img[:,:3,:,:]
		b = torch.squeeze(b).permute(1,2,0)
		plt.subplot(1,3,1)
		plt.imshow(b)
		plt.subplot(1,3,2)
		plt.imshow(a)
		plt.subplot(1,3,3)
		plt.imshow(torch.squeeze(mask))		
		plt.show()




			


