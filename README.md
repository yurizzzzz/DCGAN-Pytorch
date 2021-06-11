# DCAGN-Pytorch
This is the implementation of DCGAN([Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)) using pytorch and we generate the mnist numbers and Anime Avatar in this project.  

![](https://img04.sogoucdn.com/app/a/100520146/f4f292315764a940a3d143164b8afaec)

# Environment
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
|   |-- MNIST
|       |-- processed
|       |   |-- test.pt
|       |   `-- training.pt
|       |-- raw
|           |-- t10k-images-idx3-ubyte
|           |-- t10k-images-idx3-ubyte.gz
|           |-- t10k-labels-idx1-ubyte
|           |-- t10k-labels-idx1-ubyte.gz
|           |-- train-images-idx3-ubyte
|           |-- train-images-idx3-ubyte.gz
|           |-- train-labels-idx1-ubyte
|           `-- train-labels-idx1-ubyte.gz
|-- face_train.py           # Train to generate Anime Avatar
|-- model                 
|   |-- dcgan.py            # DCGAN Net
|-- preprocess.py           # Resize the dataset images
|-- generate_gif.py         # Generate gif
|-- test.py
|-- train.py                # Train to generate numbers
```

# Quick Start

