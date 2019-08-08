### Chest X-Ray Images (Pneumonia)

* Identify Pneunomia
  + https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

### Introduction

Deep neural networks are now the state-of-the-art machine learning models. An artificial 
intelligence system using transfer learning (architecture of one model & addition to your own created layers) can effectively classify images for diseases
and tumors. Convolutional Neural Networks (CNN) are of special interest in this field. By exploiting local connectivity patterns (colors, shapes), these networks
are able to learn to identify simple cases like animals to complex and difficult areas like cancer. The field is growing and I am super excited in what I can continue
to learn in the deep learning field.

Medical diagnosis (biomedical imaging) using machine learning and deep learning has been extremely prevalent in recent years especially in the medical industry with one 
of the latest models developed by Google. DeepMind (AI) can predict acute kidney injury before doctors that.The potential uses of artificial intelligence 
benefiting society are vast and may now include predicting who is at risk for a disease that kills nearly 2 million people worldwide every year.

Being able to accurately and quickly predict any dangerous presence of diseases early on is a very powerful tool to have. The goal of this specific
project is to create a deep learning model that is able to detect Peunomia infections in thoracic X-rays.
_____________________________________________________________________________________________

### Framework 

**TensorFlow / Keras**

### Architecture / Methods

** Still Testing **

#### Illustrative Example of a Chest X-Ray in Patients with Pneunomia
![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneunomia-radiograph-imaging/etc/xray.png)

Figure S6. Illustrative Examples of Chest X-Rays in Patients with Pneumonia, Related to Figure 6 The normal chest X-ray (left panel) 
depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal 
lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse 
"interstitial" pattern in both lungs. - [Full Text Here](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

#### Illustrative Example of Chest X-Ray Labeled with Pneunomia

TODO
_____________________________________________________________________________________________

### Data

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 
5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). Chest X-ray images (anterior-posterior) were selected from retrospective 
cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray 
imaging was performed as part of patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were initially 
screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert 
physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked 
by a third expert.
_____________________________________________________________________________________________

### Illustrative Transfer Learning Example 

![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneunomia-radiograph-imaging/etc/transfer-learning.jpg)

Figure 1. Schematic of a Convolutional Neural Network
Schematic depicting how a convolutional neural network trained on the ImageNet dataset of 1,000 categories can be adapted to significantly increase 
the accuracy and shorten the training duration of a network trained on a novel dataset of OCT images. The locally connected (convolutional) layers are 
frozen and transferred into a new network, while the final, fully connected layers are recreated and retrained from random initialization on top of the 
transferred layers.
_____________________________________________________________________________________________

### Getting Started

1. Create a conda environment ```conda create --name tf-test python=3.6``` and then load up the environment by ```conda env create -f environment.yml```.

2. If you want to utilize a Jupyter Notebook then you can do this ```python -m pykernel install --user --name tf-test --display-name "Python 3.6 (TF)"```.

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
``
_____________________________________________________________________________________________

### Contact me!

I always welcome feedback and I enjoy connecting with individuals so feel free to drop by my [LinkedIn](https://www.linkedin.com/in/davidtly) and connect!
