import tensorflow as tf 
import numpy as np 
import collections

class StyleTransfer:

	def __init__(self, content_layer_ids, style_layer_ids, init_image, content_image, style_image,
				session, net, num_iter, loss_ratio, content_loss_norm_type):

		self.net = net
		self.sess = session

		self.CONTENT_LAYERS = collections.OrderedDict(sorted(content_layer_ids.items()))
		self.STYLE_LAYERS = collections.OrderedDict(sorted(style_layer_ids.items()))

		# Preprocess
		self.p0 = np.float32(self.net.preprocess(content_image))
		self.a0 = np.float32(self.net.preprocess(style_image))
		self.x0 = np.float32(self.net.preprocess(init_image))

		# Parameters for optimization
		self.content_loss_norm_type = content_loss_norm_type
		self.num_iter = num_iter
		self.loss_ratio = loss_ratio

		self._build_graph()


	def _gram_matrix(self, tensor):
		shape = tensor.get_shape()
		num_channels = int(shape[3])
		matrix = tf.reshape(tensor, shape=[-1,num_channels])
		gram = tf.matmul(tf.transpose(matrix), matrix)
		return gram


	def _build_graph(self):
		self.x = tf.Variable(self.x0, trainable=True, dtype=tf.float32)

		self.p = tf.placeholder(tf.float32, shape=self.p0.shape, name='content')
		self.a = tf.placeholder(tf.float32, shape=self.a0.shape, name='style')

		content_layers = self.net.feed_forward(self.p, scope='content')
		self.Ps = {}
		for id in self.CONTENT_LAYERS:
			self.Ps[id] = content_layers[id]

		style_layers = self.net.feed_forward(self.a, scope='style')
		self.As = {}
		for id in self.STYLE_LAYERS:
			self.As[id] = self._gram_matrix(style_layers[id])

		self.Fs = self.net.feed_forward(self.x, scope='mixed')

		Loss_content = 0
		Loss_style = 0
		for id in self.Fs:
			if id in self.CONTENT_LAYERS:
				F = self.Fs[id] # content feature of x
				P = self.Ps[id] # content feature of p

				_, h, w, d = F.get_shape() # first return value is batch size(must be one)
				N = h.value * w.value
				M = d.value # number of filters

				w = self.CONTENT_LAYERS[id]

				if self.content_loss_norm_type==1:
					Loss_content += w * tf.reduce_sum(tf.pow((F-P),2))/2
				elif self.content_loss_norm_type==2:
					Loss_content += w * tf.reduce_sum(tf.pow((F-P),2))/(N*M)

				elif self.content_loss_norm_type==3:
					Loss_content += w * (1. / (2. * np.sqrt(M) * np.sqrt(N))) * tf.reduce_sum(tf.pow((F-P),2))

			elif id in self.STYLE_LAYERS:
				F = self.Fs[id]

				_,h,w,d = F.get_shape()
				N = h.value * w.value
				M = d.value

				w = self.STYLE_LAYERS[id]
				G = self._gram_matrix(F)
				A = self.As[id]

				Loss_style += w * (1. / (4. * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G-A),2))

		alpha = self.loss_ratio
		beta = 1

		self.Loss_content = Loss_content
		self.Loss_style = Loss_style
		self.Loss_total = alpha*Loss_content + beta*Loss_style

	def update(self):
		# define optimizer L-BFGS
		global iteration
		iteration = 0
		def callback(tl, cl, sl):
			global iteration
			print('iteration: %4d, '%iteration, 'Loss_total: %g, Loss_content: %g, L_style: %g' % (tl, cl, sl))
			iteration += 1

		optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.Loss_total, method='L-BFGS-B', options={'maxiter':self.num_iter})

		self.sess.run(tf.global_variables_initializer())

		optimizer.minimize(self.sess, feed_dict={self.a:self.a0, self.p:self.p0},
			fetches=[self.Loss_total, self.Loss_content, self.Loss_style], loss_callback=callback)

		final_image = self.sess.run(self.x)
		final_image = np.clip(self.net.unpreprocess(final_image), 0.0, 255.0)

		return final_image

