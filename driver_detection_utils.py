from sklearn.utils import shuffle
import pandas as pd 
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import random

def gen_batch_function(train_images, train_labels):
	"""
	Construct a generator to pass batch_size images and labels for training
	:images: list of all images in dataset
	:emotion: list of all corresponding ground truth labels
	"""
	print("Generator made...")

	def batch_generator(batch_size=32):

		x, y = shuffle(train_images, train_labels)
		for batch in range(0, len(x), batch_size):
			batch_x, batch_y = [], []

			for i, l in zip(x[batch:batch+batch_size], y[batch:batch+batch_size]):
				i = np.reshape(i, [i.shape[0], i.shape[1], 1])
				batch_x.append(i)
				batch_y.append(l)

			yield np.array(batch_x), np.array(batch_y)

	return batch_generator

def load_train_data():
	start_time = time.time()
	label_paths = glob.glob('./train/*')
	label = 0
	train_images, train_labels = [], []
	for path in label_paths:
		for image in glob.glob(path + '/*'):
			
			img = cv2.imread(image)
			img = preprocess(img)
			train_images.append(img)
			train_labels.append(label)

		label += 1

	train_labels = pd.get_dummies(train_labels).values #as_matrix()
	print("Data Loaded in {} second".format(time.time() - start_time))
	return train_images, train_labels

def load_test_data(num_examples=1):
	'''
	Returns list of tst images for prediction
	:num_examples: number of test images to be predicted
	'''
	imgs = glob.glob('./test/*')
	rand_imgs = random.sample(imgs, num_examples)
	test_imgs = []
	for i in range(len(rand_imgs)):
		img = cv2.imread(rand_imgs[i])
		test_imgs.append(img)

	return test_imgs

def preprocess(img):
	img = img[::3, ::3, 0]  # downsample by factor of 2
	img = cv2.resize(img, (160, 120))
	img = img[:, 20:]
	return img