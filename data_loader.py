from glob import glob
import numpy as np
from PIL import Image
import os
import math
import cv2


class DataLoader:
	def __init__(self, dataset_name, input_type, output_type, img_res_in=(64, 64), img_res_out=(32, 32)):
		self.dataset_name = dataset_name
		self.output_type = output_type
		self.img_res_in = img_res_in
		self.img_res_out = img_res_out
		self.paths_train = glob('./training_data/%s/%s/%s/*' % (dataset_name, input_type, "train"))
		self.paths_test = glob('./training_data/%s/%s/%s/*' % (dataset_name, input_type, "val"))
		self.n_batches = 1
		self.scale = 5.2  # ratio between input pixel size and training set pixel size

	def load_data(self, batch_size=1, is_testing=True):
		batch_images = np.random.choice(self.paths_test, size=batch_size)
		imgs_in, imgs_out = [], []
		
		for img_path in batch_images:		
			img_in, img_out = self.process_batch(img_path, is_testing, data_type="val")
			imgs_in.append(img_in)
			imgs_out.append(img_out)
		imgs_in = np.array(imgs_in)
		imgs_out = np.array(imgs_out)

		return imgs_in, imgs_out

	def load_batch(self, batch_size=1, is_testing=False):
		self.n_batches = int(len(self.paths_train) / batch_size)

		for i in range(self.n_batches-1):
			batch = self.paths_train[i*batch_size:(i+1)*batch_size]
			imgs_in, imgs_out = [], []
			for img_path in batch:
				img_in, img_out = self.process_batch(img_path, is_testing, data_type="train")
				imgs_in.append(img_in)
				imgs_out.append(img_out)
			imgs_in = np.array(imgs_in)
			imgs_out = np.array(imgs_out)
			yield imgs_in, imgs_out

	def process_batch(self, fn, is_testing, data_type):
		# load grayscale image
		img_in = np.expand_dims(np.array(Image.open(fn)), axis=-1)
		
		# load corresponding elastic modulus map
		path, fn = os.path.split(fn)
		path_out = os.path.join('./training_data/%s/%s/%s/%s' % (self.dataset_name, self.output_type, data_type, fn))
		img_out = np.expand_dims(np.array(Image.open(path_out)), axis=-1)
		
		# data augmentation
		img_in, img_out = data_augmentation(img_in, img_out, is_testing)

		# convert to tensor
		img_in = np.expand_dims(np.resize(img_in, self.img_res_in), axis=-1)
		img_out = np.expand_dims(np.resize(img_out, self.img_res_out), axis=-1)
		
		# adjust for tanh activation later (+/-1)
		img_in, img_out = input2tanh(img_in, img_out)
		
		return img_in, img_out

	def scale_and_pad(self, img, nr, nc):
		# scaling and padding
		ny, nx = img.shape
		img_rows_in, img_cols_in = self.img_res_in
		nx_scaled = math.ceil(self.scale * nx)
		ny_scaled = math.ceil(self.scale * ny)
		img_scaled = cv2.resize(img, (nx_scaled, ny_scaled))
		nx_pad = (img_rows_in * nc) * math.ceil(nx_scaled / (img_rows_in * nc))
		ny_pad = (img_cols_in * nr) * math.ceil(ny_scaled / (img_cols_in * nr))
		img_pad = np.zeros((ny_pad, nx_pad), dtype=np.uint8)
		img_pad[:img_scaled.shape[0], :img_scaled.shape[1]] = img_scaled

		return img_pad

	def scale_and_pad_inv(self, img_pred, ny, nx):
		# scaling and padding
		nx_scaled = math.ceil(self.scale * nx)
		ny_scaled = math.ceil(self.scale * ny)
		img_scaled = img_pred[:ny_scaled, :nx_scaled]
		img = cv2.resize(img_scaled, (nx, ny), interpolation=cv2.INTER_LINEAR)

		return img


def data_augmentation(img_in, img_out, is_testing=False):
	val = np.random.random()
	if not is_testing and 0.75 > val >= 0.5:
		img_in = np.fliplr(img_in)
		img_out = np.fliplr(img_out)
	if not is_testing and val >= 0.75:
		img_in = np.flipud(img_in)
		img_out = np.flipud(img_out)
	if not is_testing and 0.25 <= val < 0.5:
		img_in = np.rot90(img_in, k=-1)
		img_out = np.rot90(img_out, k=-1)
	if not is_testing and 0 <= val < 0.25:
		img_in = np.rot90(img_in, k=1)
		img_out = np.rot90(img_out, k=1)

	return img_in, img_out


def input2tanh(img_in, img_out):
	img_in = img_in/127.5 - 1.  # 8 bit grayscale image
	img_out = img_out/5000 - 1.  # max 10 kPa elastic modulus

	return img_in, img_out


def tanh2output(pred):
	pred = (pred + 1)*5000  # convert tanh prediction to elastic modulus

	return pred
