### OpenCV Python

* OpenCV
  + https://www.youtube.com/watch?v=kdLM6AOd2vc&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K
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