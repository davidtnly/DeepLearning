### Deep Learning with OpenCV Coursework

* OpenCV
  + https://www.linkedin.com/learning/introduction-to-deep-learning-with-opencv/next-steps

_____________________________________________________________________________________________

### Models Used

**Caffe** - https://arxiv.org/pdf/1408.5093.pdf

**YOLOv3** - https://pjreddie.com/media/files/papers/YOLOv3.pdf

YOLOv3 is the latest variant of a popular object detection algorithm YOLO – You Only Look Once. The published 
model recognizes 80 different objects in images and videos, but most importantly it is super fast and nearly 
as accurate as Single Shot MultiBox (SSD). We will be using YOLOv3 with OpenCV Implementation.

#### Sample Classification YOLOv3 Output
![Image](https://raw.githubusercontent.com/davidtnly/DeepLearning/master/03-opencv4-course/Images/fruit_YOLOv3_output.jpg)

_____________________________________________________________________________________________

### Environment

#### Create a conda environment for OpenCV

```
# Create environment
conda create --name ocv4 python=3.6
conda activate

# Install dependencies by individually installing
pip install cmake
pip install numpy
pip install pandas
pip install opencv-contrib-python-4.0.1.24
pip install ipykernel

# Create Kernel
python -m pykernel install --user --name ocv4 --display-name "Python 3.6 (Open CV)"
```

#### pip
You can also install these dependencies with pip after creating the conda environment, you can issue ```pip install -r requirements.txt```.

_____________________________________________________________________________________________

### Contact me!

I always welcome feedback and I enjoy connecting with individuals so feel free to drop by my [LinkedIn](https://www.linkedin.com/in/davidtly) and connect!
