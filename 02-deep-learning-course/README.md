### Deep Learning and Neural Networks

* Deep Learning
  + https://www.linkedin.com/learning/paths/advance-your-skills-in-deep-learning-and-neural-networks

_____________________________________________________________________________________________

### Coursework

1. Building Deep Learning Applications with Keras 2.0 - Completed July 11, 2019

2. Deep Learning: Face Recognition - Completed July 14, 2019

3. Deep Learning: Image Recognition - Completed July 15, 2019

4. Building and Deploying Deep Learning Applications with TensorFlow - Completed July 18, 2019

5. Neural Networks and Convolutional Neural Networks Essential Training - Completed July 20, 2019

6. Learning TensorFlow with JavaScript - Completed July 20, 2019 (Skimmed)

7. Accelerating TensorFlow with the Google Machine Learning Engine - Completed July 22, 2019 (Skimmed a few repetitive parts)

8. Introduction to AWS DeepLens - Completed July 22, 2019 (Skimmed)

9. NLP with Python for Machine Learning Essential Training (Not Started)

### Facial Recognition Output
![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/02-deep-learning-course/02-face-recognition/Images/facial_detection_hs.png)

### Image Classification w/ Likelihood
![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/02-deep-learning-course/02-image-recognition/Images/training.png)

_____________________________________________________________________________________________

### Environment

#### Create a conda environment for OpenCV

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

#### Requirements File
You can check the full list of dependencies in the "requirements" file if any errors occur, but do not use it to install.

_____________________________________________________________________________________________

### Contact me!

I always welcome feedback and I enjoy connecting with individuals so feel free to drop by my [LinkedIn](https://www.linkedin.com/in/davidtly) and connect!
