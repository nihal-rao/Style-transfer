import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt
import tensorflow as tf


from style_transfer.image_utils import load_image, preprocess_image, deprocess_image
from style_transfer.squeezenet import SqueezeNet
from style_transfer.loss_fns import content_loss, style_loss, tv_loss, gram_matrix


# Older versions of scipy.misc.imresize yield different results
# from newer versions, so we check to make sure scipy is up to date.
def check_scipy():
	import scipy
	version = scipy.__version__.split('.')
	if int(version[0]) < 1:
		assert int(version[1]) >= 16, "You must install SciPy >= 0.16.0 to complete this notebook."



def extract_features(x, cnn):
	"""
	Use the CNN to extract features from the input image x.
	
	Inputs:
	- x: A Tensor of shape (N, H, W, C) holding a minibatch of images that
	  will be fed to the CNN.
	- cnn: A Tensorflow model that we will use to extract features.
	
	Returns:
	- features: A list of feature for the input images x extracted using the cnn model.
	  features[i] is a Tensor of shape (N, H_i, W_i, C_i); recall that features
	  from different layers of the network may have different numbers of channels (C_i) and
	  spatial dimensions (H_i, W_i).
	"""
	features = []
	prev_feat = x
	for i, layer in enumerate(cnn.net.layers[:-2]):
		next_feat = layer(prev_feat)
		features.append(next_feat)
		prev_feat = next_feat
	return features

def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
				   style_layers, style_weights, tv_weight, result_save_path, init_random = False):
	"""Run style transfer!
	
	Inputs:
	- content_image: filename of content image
	- style_image: filename of style image
	- image_size: size of smallest image dimension (used for content loss and generated image)
	- style_size: size of smallest style image dimension
	- content_layer: layer to use for content loss
	- content_weight: weighting on content loss
	- style_layers: list of layers to use for style loss
	- style_weights: list of weights to use for each layer in style_layers
	- tv_weight: weight of total variation regularization term
	- init_random: initialize the starting image to uniform random noise
	- result_save_path: path to save the result of style transfer
	"""
	# Extract features from the content image
	content_img = preprocess_image(load_image(content_image, size=image_size))
	feats = extract_features(content_img[None], model)
	content_target = feats[content_layer]
	
	# Extract features from the style image
	style_img = preprocess_image(load_image(style_image, size=style_size))
	s_feats = extract_features(style_img[None], model)
	style_targets = []
	# Compute list of TensorFlow Gram matrices
	for idx in style_layers:
		style_targets.append(gram_matrix(s_feats[idx]))
	
	# Set up optimization hyperparameters
	initial_lr = 3.0
	decayed_lr = 0.1
	decay_lr_at = 180
	max_iter = 200
	
	step = tf.Variable(0, trainable=False)
	boundaries = [decay_lr_at]
	values = [initial_lr, decayed_lr]
	learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

	# Passing the step
	learning_rate = learning_rate_fn(step)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		
	# Initialize the generated image and optimization variables
	
	f, axarr = plt.subplots(1,2)
	axarr[0].axis('off')
	axarr[1].axis('off')
	axarr[0].set_title('Content Source Img.')
	axarr[1].set_title('Style Source Img.')
	axarr[0].imshow(deprocess_image(content_img))
	axarr[1].imshow(deprocess_image(style_img))
	plt.show()
	plt.figure()
	
	# Initialize generated image to content image
	if init_random:
		initializer = tf.random_uniform_initializer(0, 1)
		img = initializer(shape=content_img[None].shape)
		img_var = tf.Variable(img)
		print("Intializing randomly.")
	else:
		img_var = tf.Variable(content_img[None])
		print("Initializing with content image.")
		
	for t in range(max_iter):
		with tf.GradientTape() as tape:
			tape.watch(img_var)
			feats = extract_features(img_var, model)
			# Compute loss
			c_loss = content_loss(content_weight, feats[content_layer], content_target)
			s_loss = style_loss(feats, style_layers, style_targets, style_weights)
			t_loss = tv_loss(img_var, tv_weight)
			loss = c_loss + s_loss + t_loss
		# Compute gradient
		grad = tape.gradient(loss, img_var)
		optimizer.apply_gradients([(grad, img_var)])
		
		img_var.assign(tf.clip_by_value(img_var, -1.5, 1.5))
			
		if t % 100 == 0:
			print('Iteration {}'.format(t))
			plt.imshow(deprocess_image(img_var[0].numpy(), rescale=True))
			plt.axis('off')
			plt.show()
	print('Iteration {}'.format(t))    
	plt.imshow(deprocess_image(img_var[0].numpy(), rescale=True))
	plt.axis('off')
	plt.show()
	imsave(result_save_path, img_var[0].numpy())


if __name__ == '__main__':

	check_scipy()
	SAVE_PATH = './squeezenet.ckpt'
	model=SqueezeNet()
	model.load_weights(SAVE_PATH)
	model.trainable=False
	params1={'content_image' : './tubingen.jpg', #Insert the file path for the content image here
			'style_image' : './the_scream.jpg', #Insert the file path for the style image here
			'image_size' : 192,
    		'style_size':224,
    		'content_layer':2,
    		'content_weight':3e-2,
    		'style_layers':[0, 3, 5, 6],
    		'style_weights':[200000, 800, 12, 1],
    		'tv_weight':2e-2,
			'result_save_path' : '/home/mancmanomyst/nn_style_transfer/result.jpg'
			}
	style_transfer(**params1)
