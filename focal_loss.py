import torch
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np

def cross_entropy(input, target, weight=None, ignore_index=255):
	target = torch.squeeze(target, dim=1)  # (B，1，H，W)-->(B，H，W)
	# ignore_index: 
	loss = F.cross_entropy(input, target, weight, reduction='sum', ignore_index=255)
	return loss


def make_one_hot(labels, classes):
	one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
	target = one_hot.scatter_(1, labels.data, 1)

	return target


# 1、Dice Loss
class DiceLoss(nn.Module):
	def __init__(self, smooth=1, ignore_index=255):
		super(DiceLoss, self).__init__()
		self.ignore_index = ignore_index  
		self.smooth = smooth

	def forward(self, output, target):
		'''
			output: (B，n_class, H,W)  
			target: (B，H，W)  
		'''
		if self.ignore_index not in range(target.min(), target.max()):
			if (target == self.ignore_index).sum() > 0:
				target[target==self.ignore_index] = target.min()

		target = make_one_hot(target.unique(dim=1), classes=output.size()[1])
		output = F.softmax(output, dim=1)

		output_flat = output.contiguous().view(-1)
		target_flat = target.contiguous().view(-1)

		intersection = (output_flat*target_flat).sum()  
		loss = 1-(2.*intersection + self.smooth) / (output_flat.sum() + target_flat.sum() + self.smooth)

		return loss


# focal loss
class Focal_Loss(nn.Module):
	'''
		借助交叉熵求解focal loss

		loss = Focal_Loss()
		input = torch.randn(2,3,5, 5, requires_grad=True)         # （B,c,H,W）
		target = torch.empty(2,5,5, dtype=torch.long).random_(3)  #  (B，H，W) 
		output = loss(input, target)
		print(output)
		alpha-每个类别的权重
	'''
	def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
		super(Focal_Loss, self).__init__()
		self.gamma = gamma
		self.size_average = size_average

		self.CE_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index, weight=alpha)

	def forward(self, output, target):
		'''
		'''
		target = torch.squeeze(target, dim=1)  # (B，1，H，W)-->(B，H，W)
		logpt = self.CE_loss(output, target)    # 1、 -1*log(p)*y; 
		pt = torch.exp(-logpt)
		loss = ((1-pt)**self.gamma) * logpt     # 2、（1-p）**gamma * [-1*log(p)*y]

		if self.size_average:
			return loss.mean()
		return loss.sum()


class Focal_Loss_z(nn.Module):
	def __init__(self, weight, gamma=2):
		super(Focal_Loss_z,self).__init__()
		self.gamma = gamma    
		self.weight = weight  

	def forward(self, preds, labels):
		eps=1e-7
		labels = make_one_hot(labels,6)
		preds = F.softmax(preds,dim=1)
		y_pred = preds.view((preds.size()[0],preds.size()[1],-1))  # (B,C,H,W)->(B,C,H*W)
		target = labels.view(y_pred.size())  # (B,C,H,W)->(B,C,H*W)
		ce = -1.*torch.log(y_pred+eps)*target        
		floss = torch.pow((1-y_pred), self.gamma)*ce  
		floss = torch.mul(floss, self.weight)         
		floss = torch.nansum(floss, dim=1)
		floss = torch.nanmean(floss)
		return floss
