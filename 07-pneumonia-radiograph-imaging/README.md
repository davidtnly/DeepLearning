### Chest X-Ray Images (Pneumonia)

* Identify Pneumonia
  + https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

### Introduction

Deep neural networks are now the state-of-the-art machine learning models. An artificial intelligence system using transfer learning (architecture of one model & addition 
to your own created layers) can effectively classify images for diseases and tumors. Convolutional Neural Networks (CNN) are of special interest in this field. By exploiting 
local connectivity patterns (colors, shapes), these networks are able to learn to identify simple cases like animals to complex and difficult areas like cancer. The field 
is growing and I am super excited in what I can continue to learn in the deep learning field.

Medical diagnosis (biomedical imaging) using machine learning and deep learning has been extremely prevalent in recent years especially in the medical industry with one 
of the latest models developed by Google. DeepMind (AI) can predict acute kidney injury before doctors that. The potential uses of artificial intelligence 
benefitting society are vast and may now include predicting who is at risk of pneumonia, which is a disease that kills nearly 2 million people worldwide every year especially
in developing nations where billions face poverty. 

The process in developing a deep neural network model is time consuming and demands enormous amounts of resources. One method to overcome this problem is to design a
deep neural network architecture that performs image classification tasks. The proposed technique is based on the convolutional neural network (CNN) algorithm which utilizes a set
of neurons (numbers) to convolve on a given image and extract features from them. Now why convolutional neural networks over regular deep learning networks? CNNs have an edge over
certain learning tasks since it creates layers and filters that is similar to a visual schema. The network looks for colors, edges, patterns, and more in every additional layer.
In this project, we use X-ray images where the network has the ability to extract abstract 2D features through learning. 

#### Chest X-Ray in Patients with Pneumonia
![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneumonia-radiograph-imaging/images-results/xray.png)

Figure S6. Illustrative Examples of Chest X-Rays in Patients with Pneumonia, Related to Figure 6 The normal chest X-ray (left panel) 
depicts clear lungs without any areas of abnormal opacification in the image. Bacterial pneumonia (middle) typically exhibits a focal 
lobar consolidation, in this case in the right upper lobe (white arrows), whereas viral pneumonia (right) manifests with a more diffuse 
"interstitial" pattern in both lungs. - [Full Text Here](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

#### Sample Images in Training

![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneumonia-radiograph-imaging/images-results/comparison.png)

The figure above is an example of some of the images used for training. The top 3 are classified as normal images and the bottom 3 are classified as pneumonia images.

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

After multiple revisions and testing, I decided to balance the testing to exactly 50% each and training dataset to split close to 50% normal and 50% pneumonia images. The total size of the training 
set is 2,720 and testing set, including the 8 validation images, is 234 x 2. I would like to have more testing images, so one method is to grab the training images (150, 150, 3) 
and convert them (150, 150) so it is usable in my current code structure.

### Preprocessing

I used several data augmentations methods to artificially create more images for the dataset. By doing this, it can help with any underfitting or overfitting issues that may occur
since there are multiple versions of a single image. It can also enhance the model's generalization ability during training. The settings used are shown below in the image.

![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneumonia-radiograph-imaging/images-results/augment-settings.png)

### Model Development Process

Tuning the model consisted of several hyperparameter adjustments as well as image size changes. I created an initial dense layer model as proceeded to add a single regular convolutional until I reached
a 5-layer model. Hyperparameter tuning has been done manually so I can get a better idea slowly on how each adjustment would change the model even though dropout would sometimes show opposite results.
Some adjustments were left as is like kernel_size = (3, 3) and pool_size = (2, 2).

Hyperparameters: batch sizes (16, 32, 64, 128), image sizes (100, 150, 200), LeakyRelu's alpha (0.2 - 0.4), dropout (0.1 - 0.7), convolutional filters (8 - 1024)

Other processes including changing the fully connected layers multiple times from 1 dense layer to a max of 3 dense layers with 1 dropout layer to 3 dropout layers. Results varied but I decided to go with one
that would have as balanced of an average score as possible that did not have a low recall score and accuracy. In the medical field, it is worse misdiagnose a patient that has pneumonia to not have it than to 
diagnose them with a case of pneumonia when they are healthy. Lives could be saved as long as they are prescribed the needed antibiotics.

### Architecture

The final architecture is a 5-layer Separable Convolutional Neural Network w/ Leaky ReLU. There are a total of 2,238,604 trainable parameters and 960 non-trainable parameters.

![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneumonia-radiograph-imaging/images-results/shape.png)

What's Leaky ReLU vs. ReLU? ReLU is an activation function that works really well with 
most models, so why the change? Since a regular ReLU suffers from a "dying ReLU" problem, which means that the function on the negative side is zero. The neuron would be stuck
on that negative side and is unlikely to be used. With Leaky ReLU, there is a small slope on the negative side. The performance increased by about 1-2% at times so I decided
to use it on my final model.

A specific separable convolution that I am using is called the Depthwise Separable Convolution. What's a separable convolutional?  It works well with kernels that cannot be "factored" into 
two smaller kernels. In Keras, we can use ```keras.layers.SeparableConv2D``` or in TensorFlow ```tf.layers.separable_conv2d```. If you make an update to use TensorFlow 2.0, it would most 
likely be ```tf.keras.layers.SeparableConv2D```. This convolution deals with spatial dimensions as well as channels. An input image can have 3 channels and after a few convolutions, an image 
may have multiple channels. The idea is to separate one convolution into two. It convolves over the channels (depth) of the image, and then by doing pointwise convolution over all channels. 
This can reduce the number of computations by a very large amount. Here is an [article](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728) that explains the concept well. 

![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneumonia-radiograph-imaging/images-results/convolution.png)

### Results

To evaluate and validate the true accuracy of the model, I have tested several iterations of 10 epochs of the same architecture since we have several dropout layers which is 
randomized. I have chosen the architecture that showed the most balanced results. After testing, I ran a 50 epoch three times (approximately ~45 minutes each) to check how much it would vary beyond 10.

50 Epoch Training Accuracy: ~ 93.12%, Testing Accuracy: ~ 86.97%, Testing F1-Score: ~ 86.94%

10 Epoch Training Accuracy: ~ 92.61%, Testing Accuracy: ~ 88.23%, Testing F1-Score: ~ 87.71%

![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneumonia-radiograph-imaging/images-results/final-50-epoch-1.png)

![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneumonia-radiograph-imaging/images-results/final-50-epoch-cm.png)
_____________________________________________________________________________________________

### Potential Improvement Using Transfer Learning

![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/07-pneumonia-radiograph-imaging/images-results/transfer-learning.jpg)

Figure 1. Schematic of a Convolutional Neural Network
Schematic depicting how a convolutional neural network trained on the ImageNet dataset of 1,000 categories can be adapted to significantly increase 
the accuracy and shorten the training duration of a network trained on a novel dataset of OCT images. The locally connected (convolutional) layers are 
frozen and transferred into a new network, while the final, fully connected layers are recreated and retrained from random initialization on top of the 
transferred layers.

Transfer learning is a powerful deep learning method that I have yet to learn so I did not get into this method yet. By building out my own architecture, reading latest work done by other researchers, and making 
my own adjustments, I was able to develop a stronger understanding of neural networks. I still have a lot to learn and will take a lot of revisiting of topics and methods to fully grasp a concept without the aid 
of a second resource. On top of that, I still need a lot of work on working with different size images and make them workable.


### Final Thoughts

Some things I wanted to try was to get the training images to be used along with the test images. When I want to use training images, I receive an error: ValueError: could not broadcast input array 
from shape (150,150,3) into shape (150,150). So I have a reshaping issue that I need to work on for image resizing practice. It's also amazing on how some of these kernels and methods could achieve around
94% consistency on their validation data, something that I would like to get try to get to again once I learn more about how to improve.
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
