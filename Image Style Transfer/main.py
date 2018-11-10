import tensorflow as tf 
import numpy as np 
import os
import argparse

import utils
import vgg19
import style_transfer

def build_parser():
	desc = " Tensorflow implementation of 'Image Style Transfer By Convolutional Neural Networks' "
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--model_path', type=str, default='./pre_trained_model', help='The directory where the pre_trained_model is')
	parser.add_argument('--content', type=str, default='./images/sunset1.jpeg', help='File path of the content image', required=True)
	parser.add_argument('--style', type=str, default='./images/starry.jpg', help='File path of the style image', required=True)
	parser.add_argument('--output', type=str, default='./images/mixed_image.jpg', help='File path of the output image', required=True)
	parser.add_argument('--loss_ratio', type=float, default=1e-3, help='Weights of content_loss relative to style_loss')
	parser.add_argument('--content_layers', nargs='+', type=str, default=['conv4_2'], help='VGG19 layers used for content loss')
	parser.add_argument('--style_layers', nargs='+', type=str, default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
						help='VGG19 layers used for style loss')
	parser.add_argument('--content_layer_weights', nargs='+', type=float, default=[1.0], help='Content loss for each content is multiplied by corresponding weight')
	parser.add_argument('--style_layer_weights', nargs='+', type=float, default=[.2,.2,.2,.2,.2],
						help='Style loss for each content is multiplied by corresponding weight')
	parser.add_argument('--initial_type', type=str, default='content', choices=['random','content','style'], help='The initial image for optimization')
	parser.add_argument('--max_size', type=int, default=512, help='The maximum width or height of input images')
	parser.add_argument('--content_loss_norm_type', type=int, default=3, choices=[1,2,3], help='Different types of normalization for content loss')
	parser.add_argument('--num_iter', type=int, default=1000, help='The number of iterations to run')

	return parser
	

def check_args(args):
	try:
		assert len(args.content_layers) == len(args.content_layer_weights)
	except:
		print('Content layers info and weights info must be matched')
		return None

	try:
		assert len(args.style_layers) == len(args.style_layer_weights)
	except:
		print('Style layers info and weight info must be matched')
		return None

	try:
		assert args.max_size>100

	except:
		print('Too small size')
		return None

	model_file_path = args.model_path + '/' + vgg19.filename

	try:
		assert os.path.exists(model_file_path)
	except:
		print('There is no %s' % model_file_path)
		return None

	try :
		size_in_KB = os.path.getsize(model_file_path)
		assert abs(size_in_KB - 534904783) < 10
	except:
		print("Check file size of 'imagenet-vgg-verydeep-19.mat' ")
		print('There are some files with the same name')
		print('pre_trained_model used here can be download from below')
		print('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat')
		return None

	try:
		assert os.path.exists(args.content)
	except:
		print('There is no %s'%args.content)
		return None

	try:
		assert os.path.exists(args.style)
	except:
		print('There is no %s'%args.style)
		return None

	# VGG19 requires input dimension to be [batch, height, width, channel]

	
def add_one_dim(image):
	shape = (1,) + image.shape
	return np.reshape(image,shape)

def main():
	
	parser = build_parser()
	args = parser.parse_args()
	check_args(args)
	if args is None:
		exit()

	model_file_path = args.model_path + '/' + vgg19.filename
	vgg_net = vgg19.VGG19(model_file_path)

	content_image = utils.load_image(args.content, max_size=args.max_size)
	style_image = utils.load_image(args.style, shape=[content_image.shape[1],content_image.shape[0]])

	if args.initial_type == 'content':
		initial_image = content_image
	elif args.initial_type == 'style':
		initial_image = style_image
	elif args.initial_type == 'random':
		initial_image = np.ranodm.normal(size=content_image.shape, scale=np.std(content_image))

	CONTENT_LAYERS = {}
	for layer, weight in zip(args.content_layers, args.content_layer_weights):
		CONTENT_LAYERS[layer] = weight

	STYLE_LAYERS = {}
	for layer, weight in zip(args.style_layers, args.style_layer_weights):
		STYLE_LAYERS[layer] = weight

	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

	st = style_transfer.StyleTransfer(session=sess,
									content_layer_ids=CONTENT_LAYERS,
									style_layer_ids=STYLE_LAYERS,
									init_image = add_one_dim(initial_image),
									content_image = add_one_dim(content_image),
									style_image = add_one_dim(style_image),
									net=vgg_net,
									num_iter = args.num_iter,
									loss_ratio = args.loss_ratio,
									content_loss_norm_type = args.content_loss_norm_type,
									)

	result_image = st.update()
	sess.close()

	shape = result_image.shape
	result_image = np.reshape(result_image, shape[1:])

	utils.save_image(result_image, args.output)

if __name__ == "__main__":
	main()
