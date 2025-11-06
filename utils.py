import torch
import torch.nn as nn

# 模型初始化
def weight_init_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1:
		nn.init.normal_(m.weight.data,0.0,0.2)
	elif classname.find('BatchNorm2d') != -1:
		nn.init.normal_(m.weight.data,0.0,0.02)
		nn.init.constant_(m.bias.data,0.0)
	elif classname.find('Linear') != -1:
		nn.init.constant_(m.weight.data,0.0)
		if m.bias is not None:
			torch.nn.init.constant_(m.bias.data,0.0)

# 单层初始化
def weight_init_model(model):
	for layer in model.modules():
		if isinstance(layer,nn.Conv2d):
			nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
			if layer.bias is not None:
				nn.init.constant_(layer.bias,val=0.0)
		elif isinstance(layer,nn.BatchNorm2d):
			nn.init.constant_(layer.weight,val=1.0)
			nn.init.constant_(layer.bias,val=0.0)
		elif isinstance(layer,nn.Linear):
			torch.nn.init.xavier_normal_(layer.weight)
			if layer.bias is not None:
				nn.init.constant_(layer.bias,val=0.0)
	return model

