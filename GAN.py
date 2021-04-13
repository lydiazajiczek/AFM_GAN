# Standard modules
import datetime
import numpy as np
import os
from PIL import Image
import cv2
import tensorflow as tf

# Own modules
from data_loader import DataLoader, tanh2output

# Keras import statements
from keras.layers import Input, Dropout, Concatenate, AveragePooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as k
sess = k.get_session()


class GAN:
	def __init__(self, dataset_name="liver", input_type="fresh", output_type="stiffness", model_name="liver"):
		# Input shape
		self.img_rows_in, self.img_cols_in = 64, 64  # size of input training patch (intensity)
		self.img_rows_out, self.img_cols_out = 32, 32  # size of output training patch (elastic modulus)
		self.channels = 1  # grayscale images
		self.img_shape_in = (self.img_rows_in, self.img_rows_in, self.channels)
		self.img_shape_out = (self.img_rows_out, self.img_rows_out, self.channels)

		# Configure data loader
		self.dataset_name = dataset_name
		self.input_type = input_type
		self.output_type = output_type
		self.model_name = model_name
		self.data_loader = DataLoader(dataset_name=self.dataset_name,
										input_type=self.input_type,
										output_type=self.output_type,
										img_res_in=(self.img_rows_in, self.img_cols_in),
										img_res_out=(self.img_rows_out, self.img_cols_out))

		# Calculate output shape of D (PatchGAN)
		patch = int(self.img_rows_out / 2**4)
		self.disc_patch = (patch, patch, 1)

		# Number of filters in the first layer of generator and discriminator
		self.gf = 16
		self.df = 16
		
		# Could also try RMSprop
		optimizer = Adam(0.0002, 0.5)

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()   
		self.discriminator.compile(loss='mse',
									optimizer=optimizer,
									metrics=['accuracy'])

		# Build the generator
		self.generator = self.build_generator()

		# Input images and their conditioning images
		img_in = Input(shape=self.img_shape_in)
		img_out = Input(shape=self.img_shape_out)

		# By conditioning on A generate a fake version of B
		fake_out = self.generator(img_in)
		
		# For the combined model we will only train the generator
		self.discriminator.trainable = False
		
		# Is input image fake or real?
		valid = self.discriminator([img_in, fake_out])
		
		self.combined = Model(inputs=[img_in, img_out], outputs=[valid, fake_out])
		self.combined.compile(loss=['mse', 'mae'],
								loss_weights=[1, 100],
								optimizer=optimizer)
		
		print(self.generator.summary())
		
	def build_generator(self):
		"""U-Net Generator"""

		def conv2d(layer_input, filters, f_size=4, bn=True):
			"""Layers used during downsampling"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if bn:
				d = BatchNormalization(momentum=0.8)(d)
			return d

		def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0.05):
			"""Layers used during upsampling"""
			u = UpSampling2D(size=2)(layer_input)
			u = Conv2DTranspose(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
			if dropout_rate:
				u = Dropout(dropout_rate)(u)
			u = BatchNormalization(momentum=0.8)(u)
			u = Concatenate()([u, skip_input])
			return u

		# Image input
		d0 = Input(shape=self.img_shape_in)

		# Downsampling - one more layer than return arm
		d1 = conv2d(d0, self.gf, bn=False)
		d2 = conv2d(d1, self.gf*2)
		d3 = conv2d(d2, self.gf*4)
		d4 = conv2d(d3, self.gf*8)
		d5 = conv2d(d4, self.gf*8)

		# Upsampling
		u1 = deconv2d(d5, d4, self.gf*8)
		u2 = deconv2d(u1, d3, self.gf*4)
		u3 = deconv2d(u2, d2, self.gf*2)
		u4 = UpSampling2D(size=2)(u3)
		
		output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

		return Model(d0, output_img)

	def build_discriminator(self):

		def d_layer(layer_input, filters, f_size=4, bn=True, dropout_rate=0.25):
			"""Discriminator layer"""
			d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if bn:
				d = BatchNormalization(momentum=0.8)(d)
			if dropout_rate:
				d = Dropout(dropout_rate)(d)			    
			return d

		img_in = Input(shape=self.img_shape_in)

		# Downsample input image
		img_in_ds = AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid')(img_in)
		
		img_out = Input(shape=self.img_shape_out)

		# Concatenate image and conditioning image by channels to produce input
		combined_imgs = Concatenate(axis=-1)([img_in_ds, img_out])

		d1 = d_layer(combined_imgs, self.df, bn=False, dropout_rate=0.25)
		d2 = d_layer(d1, self.df*2, dropout_rate=0.25)
		d3 = d_layer(d2, self.df*4, dropout_rate=0.25)
		d4 = d_layer(d3, self.df*8, dropout_rate=0.25)
		
		validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
	
		return Model([img_in, img_out], validity)
		
	def train(self, epochs, batch_size):
	
		start_time = datetime.datetime.now()
		
		# Adversarial loss ground truths
		valid = 0.9*np.ones((batch_size,) + self.disc_patch)  # label smoothing
		fake = np.zeros((batch_size,) + self.disc_patch)
		
		for epoch in range(epochs):
			for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

				# ---------------------
				#  Train Discriminator
				# ---------------------

				fakes_B = self.generator.predict(imgs_A)
				
				# Train the discriminators (original images = real / generated = Fake)
				d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
				d_loss_fake = self.discriminator.train_on_batch([imgs_A, fakes_B], fake)
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
				
				# -----------------
				#  Train Generator
				# -----------------

				g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_B])

				elapsed_time = datetime.datetime.now() - start_time

				print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch,
						epochs,	batch_i, self.data_loader.n_batches,
						d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time))

			self.generator.save_weights("%s.h5" % self.model_name)

			# randomly sample from validation set and generate image
			# outputs two-panel image of generated and real image
			# uncomment if tweaking network architecture to see if
			# generator is converging to something useful
			# self.sample_images(epoch, batch_i=0)
			
	def sample_images(self, epoch, batch_i):
		os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
		imgs_A, imgs_B = self.data_loader.load_data(batch_size=1, is_testing=True)
		
		# convert from tanh
		fake_B = tanh2output(self.generator.predict(imgs_A))
		imgs_B = tanh2output(imgs_B)
		
		# convert to three panel image for testing
		gen_imgs = np.concatenate([np.squeeze(fake_B, axis=None), np.squeeze(imgs_B, axis=None)], axis=1)
		im = Image.fromarray(np.squeeze(gen_imgs, axis=None), mode='F')
		im.save("images/%s/%d_%d.tiff" % (self.dataset_name, epoch, batch_i))

	def predict(self, fn, nr, nc):
		start_time = datetime.datetime.now()

		self.generator.load_weights('%s.h5' % self.model_name)
		img = cv2.imread('%s.tiff' % fn, cv2.IMREAD_GRAYSCALE)

		# scale and pad
		img_pad = self.data_loader.scale_and_pad(img, nr, nc)

		# create contrast enhancement object
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(self.img_cols_out, self.img_rows_out))

		# divide input image into substacks to save memory
		img_stack = []
		for subimage in np.vsplit(img_pad, nr):
			img_stack.append(np.hsplit(subimage, nc))
		img_stack = np.reshape(img_stack, (nr * nc, int(img_pad.shape[0] / nr), int(img_pad.shape[1] / nc)))
		img_pred_stack = np.zeros(img_stack.shape, dtype=np.single)

		# process each substack
		for i in range(img_stack.shape[0]):
			print("processing subimage %d of %d" % (i+1, img_stack.shape[0]))
			img_pred = self.process_subimage(img_stack[i, :, :], clahe)
			img_pred_stack[i, :, :] = img_pred[:img_stack[i, :, :].shape[0], :img_stack[i, :, :].shape[1]]

		# concatenate results
		rows = []
		for j in range(nr):
			row = img_pred_stack[(j * nc):(j + 1) * nc, :, :]
			row = np.concatenate(row, axis=1)
			rows.append(row)
		img_pred_scaled = np.concatenate(np.asarray(rows), axis=0)
		img_pred = self.data_loader.scale_and_pad_inv(img_pred_scaled, img.shape[0], img.shape[1])
		cv2.imwrite(fn + '_pred.tiff', img_pred)

		elapsed_time = datetime.datetime.now() - start_time

		print("Total time for prediction: %s" % elapsed_time)

	def process_subimage(self, subimg, clahe):
		# apply contrast enhancement and scale
		subimg = np.array(clahe.apply(subimg) / 127.5 - 1)

		# create tensor and pad edges
		subimg_pad_tensor = np.expand_dims(
			np.expand_dims(
				np.pad(subimg, ((self.img_rows_out, self.img_rows_out), (self.img_cols_out, self.img_cols_out)),
						mode='constant'),
				axis=-1),
			axis=0)

		# extract overlapping patches
		subimg_patches = tf.extract_image_patches(subimg_pad_tensor,
														ksizes=[1, self.img_rows_in, self.img_cols_in, 1],
														strides=[1, self.img_rows_out, self.img_cols_out, 1],
														rates=[1, 1, 1, 1],
														padding='VALID')

		sp = sess.run(subimg_patches)
		# process patches
		rows = []
		for m in range(subimg_patches.shape[1]):
			row = []
			for n in range(subimg_patches.shape[2]):
				# convert patch to tensor
				patch = np.expand_dims(
							np.expand_dims(
								np.reshape(sp[0][m][n], (self.img_rows_in, self.img_cols_in)),
								axis=-1),
							axis=0)
				# predict on patch
				patch_pred = np.squeeze(
								np.squeeze(
									(self.generator.predict(patch) + 1) * 5000,
									axis=0),
								axis=-1)
				row.append(patch_pred)
			rows.append(np.hstack(row))
		img_pred = np.vstack(rows)

		return img_pred
