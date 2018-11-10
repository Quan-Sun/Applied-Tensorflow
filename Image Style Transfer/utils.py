import matplotlib.pyplot as plt 
import numpy as np
import PIL

"""Helper-function for image manipulation"""
# This function loads an image and returns it as a numpy array of floating-points
# The image can be automatically resized so the largest of the height or width is equal to max_size or the given shape

def load_image(filename, shape=None, max_size=None):
	image = PIL.Image.open(filename)

	if max_size:
		factor = float(max_size)/np.max(image.size)
		size = np.array(image.size) * factor

		# PIL requires the size to be integers
		size = size.astype(int)

		# Resize the image
		image = image.resize(size, PIL.Image.LANCZOS) # PIL.Image.LANCZOS is a resampling filter

	if shape:
		image = image.resize(shape, PIL.Image.LANCZOS)

	image = np.float32(image)
	return image

# Save images as files of *.jpeg
def save_image(image,filename):
	image = np.clip(image, 0.0, 255.0)

	image = image.astype(np.uint8) # convert float to bytes

	with open(filename, 'wb') as f:
		PIL.Image.fromarray(image).save(f, 'jpeg')


# DRAW the content-, mixed-, style-images
def draw_images(content_image, style_image, mixed_imag):
	fig,axes = plt.subplots(1,3,figsize=(10,10))

	fig.subplots_adjust(hspace=0.1,wspace=0.1)

	ax = axes.flat[0]
	ax.imshow(content_image/255.0, interpolation='sinc')
	ax.set_xlabel('Content')
	ax.set_xticks([])
	ax.set_yticks([])

	ax = axes.flat[1]
	ax.imshow(mixed_imag/255.0, interpolation='sinc')
	ax.set_xlabel('Output')
	ax.set_xticks([])
	ax.set_yticks([])

	ax = axes.flat[2]
	ax.imshow(style_image/255.0, interpolation='sinc')
	ax.set_xlabel('Style')
	ax.set_xticks([])
	ax.set_yticks([])

	plt.show()
