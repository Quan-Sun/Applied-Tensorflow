# Learning_Tensorflow
This repository will contain basic Tensorflow tutorials and deep learning concepts. It'll also contain some experiments on some papers that I read and some interesting model that I find on Internet or books.

**[Basic](https://github.com/Quan-Sun/Learning_Tensorflow/tree/master/Basic)** - A file contains basic Tensorflow tutorials, such as graph, session, tensor, nerual networks, CNN, etc. 

**[CNNs with Noisy Labels](https://github.com/Quan-Sun/Learning_Tensorflow/blob/master/CNNs%20with%20Nosiy%20Labels.ipynb)** - A notebook for an interesting experiment that I found on Github, which tests how convolutional neural networks that are trained on random labels (with some probability) are still able to acheive good accuracy on MNIST. However, my result shows a different conclusion that the random labels can surely affect the CNNs' performance.

**[CIFAR10](https://github.com/Quan-Sun/Learning_Tensorflow/tree/master/CIFAR10)** - A file contaions models for CIFAR-10 classification. CIFAR-10 classification is a common benchmark problem in machine learning. The problem is to classify RGB 32x32 pixel images across 10 categories:airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

**[Universal Approximation Theorem](https://github.com/Quan-Sun/Learning_Tensorflow/blob/master/Universal%20Approximation%20Theorem.ipynb)** - The Universal Approximation Theorem states that any feed forward neural network with a single hidden layer containing a finite number of neurons can fit any function. This notebook Creates a neural networks with one hidden layer to classifies all samples in MNIST. I respectively set 20 and 1000 hidden neurons, and test the performance of models.

**[MNIST Classification by Tensorflow_Slim](https://github.com/Quan-Sun/Learning_Tensorflow/blob/master/MNIST%20Classification%20by%20Tensorflow_Slim.ipynb)** - A notebook for constructing models by TF-Slim.

**[Image Style Transfer](https://github.com/Quan-Sun/Learning_Tensorflow/tree/master/Image%20Style%20Transfer)** - A tensorflow implementation of style transfer described in the papers **[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)**. It's a very interesting experiment. changing the parameters args_content and args_style in the last cell of notebook to produce your intereting images.
![mixed_image](https://github.com/Quan-Sun/Learning_Tensorflow/blob/master/Image%20Style%20Transfer/images/sunset1_starry.jpg)
