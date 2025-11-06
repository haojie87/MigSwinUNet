# MigSwinUNet



![](F:\code\MigSwinUNet\MigSwinUNet.png)



### **What is this repository for?**

------

MigSwinUNet is the code implementation for the paper *"**Analysis of Meandering River Migration Modes Using Remote Sensing Semantic Segmentation**"*.

------



### Who do I talk to?

------

Yu Sun； 
a. School of Earth Sciences, Northeast Petroleum University, Daqing 163318, China;
b. National Key Laboratory of Continental Shale Oil, Northeast Petroleum University, Daqing, Heilongjiang 163318, China

E-mail: sunyu_hc@163.com.

------



### Usage

------

1. Download the **MeanderSeg** dataset from https://doi.org/10.5281/zenodo.15869836.
2. Place the downloaded images and labels into the `data\\imgs` and `data\\masks` folders, respectively.
3. Set the hyperparameters in `setting.py.`
4. Run `train.py` to train the **MigSwinUNet** model.
5. Run `test.py` to evaluate the river segmentation performance of the trained model.

------



### 📁 Project Structure

------

1. **`network.py`**
    Defines the architecture of the **MigSwinUNet** model.
2. **`train.py`**
    Training script for model training.
3. **`setting.py`**
    Contains configurable hyperparameters and training settings.
4. **`data_set.py`**
    Implements the dataset loader for training and testing.
5. **`focal_loss.py`**
    Implementation of the focal loss function.
6. **`utils.py`**
    Utility functions, including model weight initialization.
7. **`test.py`**
    Script for evaluating the trained model on test data.
8. **`data/`**
   - `data/imgs`: Remote sensing images used for training
   - `data/masks`: Corresponding semantic segmentation labels

------







