import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import json
import os
import sys
import matplotlib.pyplot as  plt
import numpy as np
from utils import *
from settings import *
from focal_loss import Focal_Loss


opt = net_config
from data_set import TUDataset
from network import MigSwinUNet
net = MigSwinUNet(img_size=512, patch_size=8, in_chans=opt['in_chans'], out_chans=opt['cls_num'],
         embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
         window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
         norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
         use_checkpoint=False, fused_window_process=False)
loss_net = Focal_Loss(alpha=torch.tensor([1.0,10.0]))
optimizer_net = torch.optim.Adam(net.parameters(),lr=opt['lr'],betas=(opt['b1'],opt['b2']), weight_decay=0.01)
net = net.to(opt['device'])
loss_net.to(opt['device'])
imgFile = opt['img_path']
labelFile = opt['label_path']
if opt['begin_epoch'] != 0:
	model_path = os.path.join(opt['root_path'],'train\\saved_models\\net_%d.pth'%(opt['begin_epoch']))
	net.load_state_dict(torch.load(model_path,map_location=opt['device'],weights_only=True))
	opt['begin_epoch'] += 1
	with open('loss.json','r') as f:
		losses = json.load(f)
else:
	net.apply(weight_init_normal)
	losses = dict()
	
transforms_img = transforms.Compose([
	transforms.ToTensor(),
	])
transforms_test = transforms.Compose([
	transforms.ToTensor(),
	])

full_dataset = TUDataset(imgFile,labelFile,transforms_img=transforms_img,isTrain=True)  
train_ratio = 0.98
test_ratio = 1 - train_ratio
train_size = int(train_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
trainLoader = DataLoader(
	train_dataset,
	batch_size=opt['batch_size'],
	shuffle=True,
	num_workers=opt['worker_num']
	)
testLoader = DataLoader(
	test_dataset,
	batch_size=1, 
	shuffle=False,
	num_workers=opt['worker_num']
	)


if __name__ == '__main__':	
	for epoch in range(opt['begin_epoch'],opt['end_epoch']+1):
		# torch.cuda.empty_cache()
		iterNG = trainLoader.__iter__()
		lenNum = len(trainLoader)
		net.train()
		loss_epoch = 0
		for i in range(0,lenNum):
			batchData = iterNG.__next__()
			img = batchData['img'].to(opt['device'])
			mask = batchData['mask'].to(opt['device'])
			outputs = net(img)
			loss_u = loss_net(outputs,mask)
			optimizer_net.zero_grad()
			loss_u.backward()
			optimizer_net.step()
			sys.stdout.write('\r [Epoch %d/%d]  [Batch %d/%d] [Loss %s]' %(epoch,opt['end_epoch'],i,lenNum,loss_u.item()))
			sys.stdout.flush()
			loss_epoch += loss_u.item()
		loss_epoch = loss_epoch/lenNum
		losses['epoch_'+str(epoch)] = loss_epoch
		if opt['need_test'] and epoch % opt['test_interval'] ==0 and epoch >= opt['test_interval']:
			save_path = os.path.join(opt['root_path'], "test_result\\epoch_"+str(epoch))
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			loss_list = [loss for loss in losses.values()]
			plt.plot(np.arange(len(loss_list)), loss_list)
			plt.title('Training Loss')
			plt.xlabel('Batch Number')
			plt.ylabel('Loss')
			plt.savefig(os.path.join(save_path,'loss.png'))
			plt.close()
			print('test begin')
			net.eval()
			all_time = 0
			with torch.no_grad():
				for i, testBatch in enumerate(testLoader):
					imgTest = testBatch['img'].to(opt['device'])
					out = net(imgTest)
					out = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
					# out = out.squeeze(0).squeeze(0).cpu().numpy()
					plt.subplot(1,2,1)
					plt.imshow(np.transpose(imgTest[:,:3,:,:].squeeze(0).cpu().numpy(), (1, 2, 0)))
					plt.axis('off')
					plt.subplot(1,2,2)
					plt.imshow(out)
					plt.axis('off')
					plt.savefig('%s\\result_%d_%d.png'%(save_path,epoch,i))
					plt.close()
		if opt['need_save'] and epoch % opt['save_interval'] == 0 and epoch >= opt['save_interval']:
			save_path = os.path.join(opt['root_path'],'saved_models')
			if not os.path.exists(save_path):
				os.makedirs(save_path, exist_ok=True)
			torch.save(net.state_dict(), "%s\\net_%d.pth" % (save_path, epoch),  _use_new_zipfile_serialization=False)
			print("save weights ! epoch = %d"%epoch)
	loss_list = [loss for loss in losses.values()]
	plt.plot(np.arange(len(loss_list)), loss_list)
	plt.title('Training Loss')
	plt.xlabel('Batch Number')
	plt.ylabel('Loss')
	plt.savefig(os.path.join(opt['root_path'],'train\\saved_models\\loss.png'))
	# plt.show()
	plt.close()
	with open('loss.json', 'w') as f:
		json.dump(losses, f)

