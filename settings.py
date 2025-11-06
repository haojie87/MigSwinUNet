import torch
import os
'''
Parameter settings. Modify the commented parameters according to actual conditions.
'''


net_config = {
	'root_path': 'F:\\code\\MigSwinUNet\\',
	'img_path':os.path.join(".", "data\\imgs"),    
	'label_path': os.path.join(".", "data\\masks"),
	'device':torch.device('cpu'),
	# 'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
	'gpu_num':1,
	'worker_num': 0,
	'batch_size':1,        
	'lr':1e-3,           
	'b1':0.9,
	'b2':0.999,
	'begin_epoch':0,	  
	'end_epoch':10,	
	'need_test':True, 
	'test_interval':5,   
	'need_save':True,      
	'save_interval':5,     
	'img_height':512,      
	'img_weight':512,      
	'cls_num':2,			
	'in_chans':4,			
	'end_with':['.tif', '.tiff'],
}

