import tensorflow as tf 
import numpy as np 
import scipy.io 
from six.moves import urllib
import os

source_url = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
data_dir = './pre_trained_model'
filename = 'imagenet-vgg-verydeep-19.mat'
def maybe_download(filename):
    if not tf.gfile.Exists(data_dir):
        tf.gfile.MakeDirs(data_dir)
    file_path = os.path.join(data_dir, filename)
    
    if not tf.gfile.Exists(file_path):
        file_path, _ = urllib.request.urlretrieve(source_url, file_path)
        
        with tf.gfile.GFile(file_path) as f:
            size = f.size()
        print('Successfully download', filename, size, 'bytes.')
    return file_path

model_filename = maybe_download(filename)

def _conv_layer(input, weights, bias,padding='SAME'):
	conv = tf.nn.conv2d(input,tf.constant(weights),strides=[1,1,1,1],padding= padding)
	h_conv = conv + bias
	
	return h_conv

def _pool_layer(input, padding='SAME'):
	h_pool = tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1],padding= padding)

	return h_pool

def preprocess(image, mean_pixel):
	return image - mean_pixel

def unpreprocess(image, mean_pixel):
	return image + mean_pixel

class VGG19:
	layers = (
		'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
		'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
		'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
		'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
		'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
		)

	def __init__(self, model_filename):
		model = scipy.io.loadmat(model_filename)

		self.mean_pixel = np.array([123.68, 116.779, 103.939]) #np.mean(model['normalization'][0][0][0], axis=(0,1)) 

		self.weights = model['layers'][0]

	def preprocess(self, image):
		return image - self.mean_pixel

	def unpreprocess(self, image):
		return image + self.mean_pixel

	def feed_forward(self, input_image, scope=None):
		net = {}
		current = input_image

		with tf.variable_scope(scope):
			for num, name in enumerate(self.layers):
				type_layer = name[:4]
				if type_layer == 'conv':
					kernels = self.weights[num][0][0][2][0][0]
					bias = self.weights[num][0][0][2][0][1]

					# matconvnet: shape of weights is [width, height, in_channels, out_channels]
					# tensorflow: shape of weights is [height, width, in_channels, out_channels]

					kernels = np.transpose(kernels, [1,0,2,3])
					bias = bias.reshape(-1)
					current = _conv_layer(current, kernels, bias)

				elif type_layer == 'relu':
					current = tf.nn.relu(current)

				elif type_layer == 'pool':
					current = _pool_layer(current)

				net[name] = current
		assert len(net) == len(self.layers)
		return net