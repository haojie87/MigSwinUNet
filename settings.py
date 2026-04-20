import torch
import os
'''
Configuration settings.
Modify the parameters below according to your actual environment and task requirements.
'''

net_config = {
    'img_path': './img',          # Directory containing training images
    'label_path': './mask',      # Directory containing label masks
    'root_path': './',			# Root directory of the project

    'device': torch.device('cpu'),                    # Computing device: CPU
    # 'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # Use GPU if available

    'gpu_num': 1,                                     # Number of GPUs to use
    'worker_num': 0,                                  # Number of data loading workers
    'batch_size': 1,                                  # Batch size
    'lr': 1e-3,                                       # Learning rate
    'b1': 0.9,                                        # Beta1 for Adam optimizer
    'b2': 0.999,                                      # Beta2 for Adam optimizer

    'begin_epoch': 0,                                 # Starting epoch
    'end_epoch': 100,                                 # Ending epoch

    'need_test': True,                                # Whether to evaluate the model during training
    'test_interval': 50,                              # Test the model every N epochs

    'need_save': True,                                # Whether to save the model during training
    'save_interval': 50,                              # Save the model every N epochs

    'img_height': 512,                                # Input image height
    'img_weight': 512,                                # Input image width
    'cls_num': 2,                                     # Number of output classes
    'in_chans': 4,                                    # Number of input channels
    'end_with': ['.tif', '.tiff'],                    # Supported image file extensions
}
