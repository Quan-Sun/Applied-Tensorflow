# Master-of-Tensorflow
This repository will contain basic Tensorflow tutorials and deep learning concepts. It'll also contain some experiments on some papers that I read and some interesting model that I find on Internet or books.

**[Basic](https://github.com/Quan-Sun/Learning_Tensorflow/tree/master/Basic)** - A file contains basic Tensorflow tutorials, such as graph, session, tensor, nerual networks, etc. 

**[MNIST_CNN](https://github.com/Quan-Sun/Applied-Tensorflow/blob/master/MNIST_CNN.ipynb)** - A notebook for MNIST classification by a simple CNN model, but getting a very high testing accuracy.

**[CNNs with Noisy Labels](https://github.com/Quan-Sun/Learning_Tensorflow/blob/master/CNNs%20with%20Nosiy%20Labels.ipynb)** - A notebook for an interesting experiment that I found on Github, which tests how convolutional neural networks that are trained on random labels (with some probability) are still able to acheive good accuracy on MNIST. However, my result shows a different conclusion that the random labels can surely affect the CNNs' performance.

**[CIFAR10](https://github.com/Quan-Sun/Learning_Tensorflow/tree/master/CIFAR10)** - A file contaions models for CIFAR-10 classification. CIFAR-10 classification is a common benchmark problem in machine learning. The problem is to classify RGB 32x32 pixel images across 10 categories:airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

**[Universal Approximation Theorem](https://github.com/Quan-Sun/Learning_Tensorflow/blob/master/Universal%20Approximation%20Theorem.ipynb)** - The Universal Approximation Theorem states that any feed forward neural network with a single hidden layer containing a finite number of neurons can fit any function. This notebook Creates a neural networks with one hidden layer to classifies all samples in MNIST. I respectively set 20 and 1000 hidden neurons, and test the performance of models.

**[MNIST Classification by Tensorflow_Slim](https://github.com/Quan-Sun/Learning_Tensorflow/blob/master/MNIST%20Classification%20by%20Tensorflow_Slim.ipynb)** - A notebook for constructing models by TF-Slim.

**[Image Style Transfer](https://github.com/Quan-Sun/Learning_Tensorflow/tree/master/Image%20Style%20Transfer)** - A tensorflow implementation of style transfer described in the paper **[Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)**. It's a very interesting experiment. changing the parameters args_content and args_style in the last cell of notebook to produce your intereting images. I recommend you train this model on colab.

<div align=center><img src="https://github.com/Quan-Sun/Learning_Tensorflow/raw/master/Image%20Style%20Transfer/images/sunset1_starry.jpg"/></div>

**[The_simplest_RNN](https://github.com/Quan-Sun/Applied-Tensorflow/blob/master/The_simplest_RNN.ipynb)** - A notebook for MNIST classification by a very simple RNN model, but getting a relatively high test accuracy.

**[models](https://github.com/Quan-Sun/Master-of-Tensorflow/tree/master/models)** - A fold contains some well-known models, such as VGG19, ResNet50 etc.


**[MNIST classification with SVM](https://github.com/Quan-Sun/Dive-into-Machine-Learning/blob/master/MNIST%20classification%20with%20SVM.ipynb)** - A notebook using classical SVM with RBF kernel to classify MNIST in scikit-learn, which launching grid search with cross-validation for finding the best parameters, C and gamma. As a result, best parameters are C=5 and gamma=0.05. I didn't complete the training process due to its long time running.
