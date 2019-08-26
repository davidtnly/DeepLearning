### Welcome to my Deep Learning repository
_____________________________________________________________________________________________

This is a testing and learning ground for all deep learning related.

* Image Clustering
  + https://github.com/davidtnly/DeepLearning/tree/master/01-image-clustering
* Deep Learning & Neural Network Course
  + https://www.linkedin.com/learning/paths/advance-your-skills-in-deep-learning-and-neural-networks
* Deep Learning Approach to Pneumonia Classification
  + https://github.com/davidtnly/DeepLearning/tree/master/07-pneumonia-radiograph-imaging
* Dogs vs. Cats Classifier
  + https://www.kaggle.com/c/dogs-vs-cats
* SIIM-ACR Pneumothorax Segmentation
  + https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

### Environment Setup (Using environment.yml)
_____________________________________________________________________________________________

1. Create a conda environment ```conda create --name tf-test python=3.6``` and then load up the environment by ```conda env create -f environment.yml```. The environment file could be found [here](https://github.com/davidtnly/DeepLearning/tree/master/07-pneumonia-radiograph-imaging).

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
Now you can open up your Jupyter notebook and select the new kernel labled as the display name you set it to.

### Environment Setup (Installing libraries)
_____________________________________________________________________________________________

```
# Create environment
conda create --name tensorflow python=3.6
activate tensorflow
conda install jupyter
conda install scipy

# Install dependencies by individually installing
pip install --upgrade sklearn
pip install --upgrade pandas
pip install --upgrade pandas-datareader
pip install --upgrade matplotlib
pip install --upgrade pillow
pip install --upgrade tqdm
pip install --upgrade requests
pip install --upgrade h5py
pip install --upgrade pyyaml
pip install --upgrade tensorflow_hub
pip install --upgrade bayesian-optimization
pip install --upgrade spacy
pip install --upgrade gensim
pip install --upgrade tensorflow==1.14.0
pip install --upgrade keras==2.2.4
pip install ipykernel

# Create Kernel
python -m ipykernel install --user --name tensorflow --display-name "Python 3.6 (TensorFlow)"
```

Now you can open up your Jupyter notebook and select the new kernel labled as the display name you set it to. You can install the same libraries if you are using an IDE like PyCharm.
_____________________________________________________________________________________________

```
import deeplearning
import study

knowledge = []
topic = deeplearning.topic('')

if topic == 'new':
    study.learn(topic)
else:
    knowledge.append(topic)
```

## Contact me!
_____________________________________________________________________________________________

I always welcome feedback and I enjoy connecting with individuals so feel free to drop by my [LinkedIn](https://www.linkedin.com/in/davidtly) and connect!
