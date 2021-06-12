# DCAGN-Pytorch
This is the implementation of DCGAN([Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)) using pytorch and we generate the mnist numbers and Anime Avatar in this project.  

![](https://img04.sogoucdn.com/app/a/100520146/f4f292315764a940a3d143164b8afaec)

# Development Environment
- pytorch 1.4
- Python 3.7
- torchvision 1.1
- CUDA 10.0
- tqdm
- NVIDIA 2080ti
- Ubuntu 18.04
- OpenCV

# Content
```
|-- data
|   |-- MNIST                 # Mnist dataset
|       |-- processed
|       |   |-- test.pt
|       |   |-- training.pt
|       |-- raw
|-- face_train.py             # train to generate the Anime Avatar
|-- generate_gif.py           # generate gif images
|-- mnist_train.py            # train to generate the mnist numbers
|-- model                   
|   |-- __pycache__
|   |   |-- dcgan.cpython-37.pyc
|   |-- face_dcgan.py         # DCGAN (3 channels)
|   |-- mnist_dcgan.py        # DCGAN (1 channel)
|-- preprocess.py             # Crop images
|-- result                    # result gif
|   |-- comic.gif
|   |-- mnist.gif
```

# Quick Start
- Train the Anime Avatar (you can modify the epochs and other args)  
  ```
  $ cd DCGAN-pytorh-master
  $ ./face_train.py
  ```
- Train the mnist numbers (you can modify the epochs and other args) 
  ```
  $ cd DCGAN-pytorh-master
  $ ./mnist_train.py
  ```

# Results
## Anime Avatar
- Generate the Anime Avatar using fixed random noise (totally 50 epochs)  
  
  ![comic](https://github.com/FanDady/DCGAN-Pytorch/blob/master/result/comic.gif)

## Mnist numbers
- Generate the mnist numbers using fixed random noise (totally 20 epochs)  
  
  ![comic](https://github.com/FanDady/DCGAN-Pytorch/blob/master/result/mnist.gif)
