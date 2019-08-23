### Dog & Cat Classification Using Transfer Learning

* Dog vs. Cat Image Classification
  + https://www.kaggle.com/c/dogs-vs-cats

### Introduction
_____________________________________________________________________________________________

Write an algorithm to classify whether images contain either a dog or a cat. Asirra (Animal Species Image Recognition for Restricting Access) is a HIP that works by asking users to identify photographs of cats and dogs. This task is difficult for computers, but studies have shown that people can accomplish it quickly and accurately. While random guessing is the easiest form of attack, various forms of image recognition can allow an attacker to make guesses that are better than random. There is enormous diversity in the photo database (a wide variety of backgrounds, angles, poses, lighting, etc.), making accurate automatic classification difficult. In an informal poll conducted many years ago, computer vision experts posited that a classifier with better than 60% accuracy would be difficult without a major advance in the state of the art. For reference, a 60% classifier improves the guessing probability of a 12-image HIP from 1/4096 to 1/459. The state of the art at the time was about 80% accuracy.

### Method
_____________________________________________________________________________________________

Utilize transfer learning to create a model that would be able to identify cats and dogs with limited data. The dataset provided consists of 12.5k images of each class to a total of 25k images. I wanted to learn and use transfer learning fast so I decided to limit my training set to 2k per class and 1k validation per class. I used a total of 40 images in the end to test out the model created by our highest scoring model. I started out using a simple convolutional neural network architecture that I was familiar with (VGG16) and started from there. I utilized a total of 4 architectures: VGG16, MobileNetV2, ResNet50, and InceptionV3. More models built within Keras can be found [here](https://github.com/keras-team/keras-applications).

### Results
_____________________________________________________________________________________________

The image below shows a few of the predicted classes from the final InceptionV3 model, which isn't that great although the training and validation accuracy was 99% - 100%.

![Image](https://github.com/davidtnly/DeepLearning/blob/master/09-dog-cat-classifier/images-results/testing-inceptionv3.png)

### What is transfer learning?
_____________________________________________________________________________________________

I mentioned that I am using transfer learning for this project. What exactly is transfer learning and why am I using it? Transfer learning is a common and highly effective approach to deep learning on image datasets by leveraging a pre-trained network. From Chapter 5, Section 3 of [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff), a pre-trained network is simply a saved network previously trained on large dataset. From [Deep Learning](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=as_li_ss_tl?_encoding=UTF8&qid=&sr=&linkCode=sl1&tag=inspiredalgor-20&linkId=e4e32749958369afb667e7e4323d65ba&language=en_US), the weights in re-used layers may be used as the starting point for the training process and adapted in response to the new problem. The objective is to take advantage of data from the first setting to extract information that may be useful when learning or even when directly making predictions in the second setting. In our examples, the networks were trained on the ImageNet dataset (hence using "imagenet" weights when building the base model). If the original dataset is large enough and general enough, meaning that it would be able to detect similar features that I am trying to do on my final classification task, tthen the spacial features learned by the pre-trained network can effectively act as a generic model. So we will be building this pre-trained network as our initial model to hopefully detect the features of the cats and dog. More on what a pre-trained convolutional net [here](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb).

There are two ways to leverage a pre-trained network: feature extraction and fine-tuning. Here's a really good example and excerpt below from Francois Chollet. You can get the full text [here](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb).

Feature extraction consists of using the representations learned by a previous network to extract interesting features from new samples. These features are then run through a new classifier, which is trained from scratch. As we saw previously, convnets used for image classification comprise two parts: they start with a series of pooling and convolution layers, and they end with a densely-connected classifier. The first part is called the "convolutional base" of the model. In the case of convnets, "feature extraction" will simply consist of taking the convolutional base of a previously-trained network, running the new data through it, and training a new classifier on top of the output.

Fine-tuning

Another widely used technique for model reuse, complementary to feature extraction, is fine-tuning. Fine-tuning consists in unfreezing a few of the top layers of a frozen model base used for feature extraction, and jointly training both the newly added part of the model (in our case, the fully-connected classifier) and these top layers. This is called "fine-tuning" because it slightly adjusts the more abstract representations of the model being reused, in order to make them more relevant for the problem at hand.

It is necessary to freeze the convolution base of VGG16 in order to be able to train a randomly initialized classifier on top. For the same reason, it is only possible to fine-tune the top layers of the convolutional base once the classifier on top has already been trained. If the classified wasn't already trained, then the error signal propagating through the network during training would be too large, and the representations previously learned by the layers being fine-tuned would be destroyed. Thus the steps for fine-tuning a network are as follow:

- Add your custom network on top of an already trained base network
- Freeze the base network
- Train the part you added
- Unfreeze some layers in the base network
- Jointly train both these layers and the part you added

Why not fine-tune more layers?

Why not fine-tune the entire convolutional base? We could. However, we need to consider that:

Earlier layers in the convolutional base encode more generic, reusable features, while layers higher up encode more specialized features. It is more useful to fine-tune the more specialized features, as these are the ones that need to be repurposed on our new problem. There would be fast-decreasing returns in fine-tuning lower layers. The more parameters we are training, the more we are at risk of overfitting. The convolutional base has 15M parameters, so it would be risky to attempt to train it on our small dataset.

Thus, in our situation, it is a good strategy to only fine-tune the top 2 to 3 layers in the convolutional base.

### Preprocessing 
_____________________________________________________________________________________________

The preprocessing for this specific project was simple. Split the data into two folders for each class and set. So we have a train and validation folder each containing a "cats" and "dogs" folder of images. The only augmentation I did was rescaling on our final model.

### Architectures
_____________________________________________________________________________________________

CNN Architectures used:

- VGG16 - this model can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels). The default input size for this model is 224x224.
- MobileNetV2 - this model only supports the data format 'channels_last' (height, width, channels). The default input size for this model is 224x224.
- ResNet50 - this model and can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels). The default input size for this model is 224x224.
- InceptionV3 - this model and can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels). The default input size for this model is 299x299.

### Model Results
_____________________________________________________________________________________________

Here we have the final epochs from the VGG fine-tuning phase. The training accuracy got up to 99% and the validation accuracy almost up to 93%.

![Image](https://github.com/davidtnly/DeepLearning/tree/master/09-dog-cat-classifier/images-results/fine-tuning-vgg.png)
![Image](https://github.com/davidtnly/DeepLearning/tree/master/09-dog-cat-classifier/images-results/fine-tuning-plot-vgg.png)

After using the VGG16 network, I wanted to use an architecture that was a little more complicated, which was also fast, MobileNetV2.

![Image](https://github.com/davidtnly/DeepLearning/tree/master/09-dog-cat-classifier/images-results/fine-tuning-mobilenet.png)
![Image](https://github.com/davidtnly/DeepLearning/tree/master/09-dog-cat-classifier/images-results/fine-tuning-plot-mobilenet.png)

ResNet50

![Image](https://github.com/davidtnly/DeepLearning/tree/master/09-dog-cat-classifier/images-results/fine-tuning-resnet.png)
![Image](https://github.com/davidtnly/DeepLearning/tree/master/09-dog-cat-classifier/images-results/fine-tuning-plot-resnet.png)

InceptionV3

![Image](https://github.com/davidtnly/DeepLearning/tree/master/09-dog-cat-classifier/images-results/fine-tuning-inceptionv3.png)
![Image](https://github.com/davidtnly/DeepLearning/tree/master/09-dog-cat-classifier/images-results/fine-tuning-plot-inceptionv3.png)

### Potential Improvement Using Transfer Learning

- More aggresive data augmentation
- More aggressive dropout
- Use of L1 and L2 regularization (also known as "weight decay")
- Fine-tuning one more convolutional blocks (alongside greater regularization)

### Final Thoughts
_____________________________________________________________________________________________

I would want to train the entire dataset and test it on at least 5k of each classes on the test dataset. I would need to label the images because the are not labeled so that's another challenge. I could make the training and validation dataset smaller to accomodate at least 500 or 1k images per class. The sample test set I did with the current 4k/2k set inception v3 model is not that great. The goal was to learn how transfer learning works and it's still a work in progress. I hope to start creating more complicated architectures as well that have high performance and results from the ImageNet challenge. This is a good baseline on what I've been learning and will build on that. So far I worked with building my own classifier in my Pneumonia classification project to utilizing transfer learning methods to develop a stronger performing model that was trained on millions of images that is considered state of the art when they were developed. The models are great regardless and I will continue to learn more that are newer as well and learn how to make even better and faster products that can be used.

### Environment Setup
_____________________________________________________________________________________________

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
### References
_____________________________________________________________________________________________

VGG16 - [paper](https://arxiv.org/abs/1409.1556)
MobileNetV2- [paper](https://arxiv.org/abs/1801.04381)
ResNet50 - [paper](https://arxiv.org/abs/1512.03385)
InceptionV3 - [paper](https://arxiv.org/abs/1512.00567)

### More on Transfer Learning
_____________________________________________________________________________________________

Deep Learning Book - [Link](https://www.deeplearningbook.org/)
TensorFlow Documentation (Which is what I used to basically ended up building my models on) - [Link](https://www.tensorflow.org/tutorials/images/transfer_learning)
MLM Article - [Link](https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/)
MobileNetV2- [Link](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
PIS Article- [Link](https://www.pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/)
Googling Resources - [Link](https://www.google.com/search?client=firefox-b-1-d&q=how+to+improve+transfer+learning+model)

### Contact me!
_____________________________________________________________________________________________

I always welcome feedback and I enjoy connecting with individuals so feel free to drop by my [LinkedIn](https://www.linkedin.com/in/davidtly) and connect!
