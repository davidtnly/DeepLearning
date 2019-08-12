### Chest X-Ray Images (Pneumonia)

* Identify Pneunomia
  + https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

### Introduction

Deep neural networks are now the state-of-the-art machine learning models. An artificial intelligence system using transfer learning (architecture of one model & addition 
to your own created layers) can effectively classify images for diseases and tumors. Convolutional Neural Networks (CNN) are of special interest in this field. By exploiting 
local connectivity patterns (colors, shapes), these networks are able to learn to identify simple cases like animals to complex and difficult areas like cancer. The field 
is growing and I am super excited in what I can continue to learn in the deep learning field.

Medical diagnosis (biomedical imaging) using machine learning and deep learning has been extremely prevalent in recent years especially in the medical industry with one 
of the latest models developed by Google. DeepMind (AI) can predict acute kidney injury before doctors that.The potential uses of artificial intelligence 
benefitting society are vast and may now include predicting who is at risk of pneumonia, which is a disease that kills nearly 2 million people worldwide every year especially
in developing nations where billions face poverty. 

The process in developing a deep neural network model is time consuming and demands enormous amounts of resources. One method to overcome this problem is to design a
deep neural network architecture that performs image classification tasks. The proposed technique is based on the convolutional neural network (CNN) algorithm which utilizes a set
of neurons (numbers) to convolve on a given image and extract features from them. Now why convolutional neural networks over regular deep learning networks? CNNs have an edge over
certain learning tasks since it creates layers and filters that is similar to a visual schema. The network looks for colors, edges, patterns, and more in every additional layer.
Using X-ray images, it has the ability to extract abstract 2D features through learning. 

#### Chest X-Ray in Patients with Pneunomia
![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneunomia-radiograph-imaging/etc/xray.png)

Figure S6. Illustrative Examples of Chest X-Rays in Patients with Pneumonia, Related to Figure 6 The normal chest X-ray (left panel) 
depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal 
lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse 
"interstitial" pattern in both lungs. - [Full Text Here](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

### Goal

Being able to accurately and quickly predict any dangerous presence of diseases early on is a very powerful tool to have. The goal of this specific project is to create a deep 
learning neural network from scratch without transfer learning that is able to detect Pneumonia infections in thoracic X-rays.

Additional areas to learn, strengthen, and improve on is know what are some of the latest papers on deep learning networks. What's the architecture behind the networks and why are
they structured the way they are. Question almost everything on why certain layers work while some don't and what are ways to improve performance of the entire model?

### Data and Methods

The original dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 
5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). Chest X-ray images (anterior-posterior) were selected from retrospective 
cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray 
imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially 
screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert 
physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked 
by a third expert.

After multiple revisions and testing, I decided to balance the testing and training dataset to split to exactly 50% normal and 50% pneumonia images. The total size of the training 
set is 1341 x 2 and testing set, including the 8 validation images, is 234 x 2. I would like to have more testing images, so one method is to grab the training images (150, 150, 3) 
and convert them (150, 150) so it is usable in my current code structure.

### Preprocessing

I used several data augmentations methods to artifically create more images for the dataset. By doing this, it can help with any underfitting or overfitting issues that may occur
since there are multiple versions of a single image. It can also enhance the model's generalization ability during training. The settings used are shown below in the image.

![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneunomia-radiograph-imaging/etc/augment-settings.png)

### Architecture

Final Model: 5-layer Separable Convolutional Neural Network w/ Leaky ReLU

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_layer (InputLayer)     (None, 150, 150, 3)       0         
_________________________________________________________________
conv1_1 (SeparableConv2D)    (None, 150, 150, 16)      91        
_________________________________________________________________
leaky_re_lu_161 (LeakyReLU)  (None, 150, 150, 16)      0         
_________________________________________________________________
conv1_2 (SeparableConv2D)    (None, 150, 150, 16)      416       
_________________________________________________________________
leaky_re_lu_162 (LeakyReLU)  (None, 150, 150, 16)      0         
_________________________________________________________________
pool1_1 (MaxPooling2D)       (None, 75, 75, 16)        0         
_________________________________________________________________
conv2_1 (SeparableConv2D)    (None, 75, 75, 32)        688       
_________________________________________________________________
leaky_re_lu_163 (LeakyReLU)  (None, 75, 75, 32)        0         
_________________________________________________________________
conv2_2 (SeparableConv2D)    (None, 75, 75, 32)        1344      
_________________________________________________________________
leaky_re_lu_164 (LeakyReLU)  (None, 75, 75, 32)        0         
_________________________________________________________________
bn2_1 (BatchNormalization)   (None, 75, 75, 32)        128       
_________________________________________________________________
pool2_1 (MaxPooling2D)       (None, 37, 37, 32)        0         
_________________________________________________________________
conv3_1 (SeparableConv2D)    (None, 37, 37, 64)        2400      
_________________________________________________________________
leaky_re_lu_165 (LeakyReLU)  (None, 37, 37, 64)        0         
_________________________________________________________________
conv3_2 (SeparableConv2D)    (None, 37, 37, 64)        4736      
_________________________________________________________________
leaky_re_lu_166 (LeakyReLU)  (None, 37, 37, 64)        0         
_________________________________________________________________
bn3_1 (BatchNormalization)   (None, 37, 37, 64)        256       
_________________________________________________________________
pool3_1 (MaxPooling2D)       (None, 18, 18, 64)        0         
_________________________________________________________________
conv4_1 (SeparableConv2D)    (None, 18, 18, 128)       8896      
_________________________________________________________________
leaky_re_lu_167 (LeakyReLU)  (None, 18, 18, 128)       0         
_________________________________________________________________
conv4_2 (SeparableConv2D)    (None, 18, 18, 128)       17664     
_________________________________________________________________
leaky_re_lu_168 (LeakyReLU)  (None, 18, 18, 128)       0         
_________________________________________________________________
bn4_1 (BatchNormalization)   (None, 18, 18, 128)       512       
_________________________________________________________________
pool4_1 (MaxPooling2D)       (None, 9, 9, 128)         0         
_________________________________________________________________
dropout4_1 (Dropout)         (None, 9, 9, 128)         0         
_________________________________________________________________
conv5_1 (SeparableConv2D)    (None, 9, 9, 256)         34176     
_________________________________________________________________
leaky_re_lu_169 (LeakyReLU)  (None, 9, 9, 256)         0         
_________________________________________________________________
conv5_2 (SeparableConv2D)    (None, 9, 9, 256)         68096     
_________________________________________________________________
leaky_re_lu_170 (LeakyReLU)  (None, 9, 9, 256)         0         
_________________________________________________________________
bn5_1 (BatchNormalization)   (None, 9, 9, 256)         1024      
_________________________________________________________________
pool5_1 (MaxPooling2D)       (None, 4, 4, 256)         0         
_________________________________________________________________
dropout5_1 (Dropout)         (None, 4, 4, 256)         0         
_________________________________________________________________
flatten6_1 (Flatten)         (None, 4096)              0         
_________________________________________________________________
dropout6_1 (Dropout)         (None, 4096)              0         
_________________________________________________________________
fc6_1 (Dense)                (None, 512)               2097664   
_________________________________________________________________
leaky_re_lu_171 (LeakyReLU)  (None, 512)               0         
_________________________________________________________________
output (Dense)               (None, 1)                 513       
=================================================================
Total params: 2,238,604
Trainable params: 2,237,644
Non-trainable params: 960
'''

### Results

Final Model: Training Accuracy: ~ 92.61%, Testing Accuracy: ~ 88.23%, Testing F1-Score: ~ 87.71%

_____________________________________________________________________________________________

### Potential Improvement Using Transfer Learning

![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneunomia-radiograph-imaging/etc/transfer-learning.jpg)

Figure 1. Schematic of a Convolutional Neural Network
Schematic depicting how a convolutional neural network trained on the ImageNet dataset of 1,000 categories can be adapted to significantly increase 
the accuracy and shorten the training duration of a network trained on a novel dataset of OCT images. The locally connected (convolutional) layers are 
frozen and transferred into a new network, while the final, fully connected layers are recreated and retrained from random initialization on top of the 
transferred layers.
_____________________________________________________________________________________________

### Environment Setup

1. Create a conda environment ```conda create --name tf-test python=3.6``` and then load up the environment by ```conda env create -f environment.yml```.

2. If you want to utilize a Jupyter Notebook then you can do this ```python -m pykernel install --user --name tf-test --display-name "Python 3.6 (TensorFlow)"```.

```
Current versions working for me

# Check DL Versions
OpenCV Version: 4.0.1
TensorFlow Version: 1.13.1
TensorFlow Keras Version: 2.2.4-tf
Keras Version: 2.2.4

# Check GPU
TensorFlow-GPU is available
TensorFlow CUDA: True
Tensorflow GPU Device Currently Activated: /device:GPU:0
Keras GPU: ['/job:localhost/replica:0/task:0/device:GPU:0']

# Check Python Version
Python 3.6.8 |Anaconda, Inc.| (default, Feb 21 2019, 18:30:04) [MSC v.1916 64 bit (AMD64)]
```
_____________________________________________________________________________________________

### Contact me!

I always welcome feedback and I enjoy connecting with individuals so feel free to drop by my [LinkedIn](https://www.linkedin.com/in/davidtly) and connect!
