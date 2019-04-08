import tensorflow as tf
from driver_detection_utils import *
import pandas as pd 
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import time
import random

MODEL = 'simple' # or vgg16
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-4
label_dict = {0: "Safe", 
					1: "Texting (Right)", 
					2: "Phone (Right)", 
					3: "Texting (Left)", 
					4: "Phone (Left)", 
					5: "Radio", 
					6: "Drinking", 
					7: "Reach Behind", 
					8: "Hair & Makeup", 
					9: "Talking"}

def main():
	X, Y = load_train_data()

	# ## Splitting Dataset. Need to shuffle first.
	X, Y = shuffle(X, Y) 
	split = int(0.90*len(X))
	x_train, y_train = X[:split], Y[:split]
	x_val, y_val = X[split:], Y[split:]
	input_shape = (120, 140)
	num_classes = 10
	batch_generator_train = gen_batch_function(train_images=x_train, train_labels=y_train)
	batch_generator_val = gen_batch_function(train_images=x_val, train_labels=y_val)

	## Construct and train Network
	cnn = CNN(num_classes=num_classes, learning_rate=LEARNING_RATE, input_shape=input_shape, model=MODEL)
	epoch_error, val_error = cnn.train(batch_size=BATCH_SIZE, epochs=EPOCHS, batch_generator_train=batch_generator_train, batch_generator_val=batch_generator_val)
	print(epoch_error, val_error)
	plt.plot(epoch_error, color='red')
	plt.plot(val_error, color='blue')
	plt.show()


############################## Models ####################################
class CNN:
	def __init__(self, num_classes, learning_rate, input_shape, model):
		self.save_path = './driver_distraction_models/'
		self.save_epoch = 1
		self.model = model
		self.num_classes=num_classes
		self.learning_rate=learning_rate
		self.num_classes = num_classes

		self.x = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], 1])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])
		self.sess = tf.InteractiveSession()

		if self.model == 'vgg16':
			self.build_vgg16(trainable=True)
		elif self.model == 'simple':
			self.build_simple()

		self.loss = tf.losses.softmax_cross_entropy(logits=self.out, onehot_labels=self.y, )
		self.error = tf.reduce_mean(self.loss)
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = optimizer.minimize(self.error)

		self.sess.run(tf.global_variables_initializer())

	
	def build_simple(self):

		layer1 = self.conv2d(self.x, num_filters=32, name='layer1', ksize=3, trainable=True)
		layer1 = tf.layers.max_pooling2d(layer1, pool_size=2, strides=2)

		layer2 = self.conv2d(layer1, num_filters=64, name='layer2', ksize=3, trainable=True)
		layer2 = tf.layers.max_pooling2d(layer2, pool_size=2, strides=2)

		layer3 = self.conv2d(layer2, num_filters=128, name='layer3', ksize=3, trainable=True)
		layer3 = tf.layers.max_pooling2d(layer3, pool_size=2, strides=2)

		dense1 = tf.contrib.layers.flatten(layer3)
		dense1 = tf.layers.dense(dense1, units=256, activation=tf.nn.relu)
		self.out = tf.layers.dense(dense1, units=self.num_classes)

	def build_vgg16(self, trainable=True):

		conv1_1 = self.conv2d(self.x, num_filters=64, name='conv1_1', trainable=trainable)
		conv1_2 = self.conv2d(conv1_1, num_filters=64, name='conv1_2', trainable=trainable)
		max_pool1 = tf.layers.max_pooling2d(conv1_2, pool_size= 2, strides= 2, padding='SAME')

		conv2_1 = self.conv2d(max_pool1, num_filters=128, name='conv2_1', trainable=trainable)
		conv2_2 = self.conv2d(conv2_1, num_filters=128, name='conv2_2', trainable=trainable)
		max_pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=2, strides=2, padding='SAME')

		conv3_1 = self.conv2d(max_pool2, num_filters=256, name='conv3_1', trainable=trainable)
		conv3_2 = self.conv2d(conv3_1, num_filters=256, name='conv3_2', trainable=trainable)
		conv3_3 = self.conv2d(conv3_2, num_filters=256, name='conv3_3', trainable=trainable)
		max_pool3 = tf.layers.max_pooling2d(conv3_3, pool_size=2, strides=2, padding='SAME')

		conv4_1 = self.conv2d(max_pool3, num_filters=512, name='conv4_1', trainable=trainable)
		conv4_2 = self.conv2d(conv4_1, num_filters=512, name='conv4_2', trainable=trainable)
		conv4_3 = self.conv2d(conv4_2, num_filters=512, name='conv4_3', trainable=trainable)
		max_pool4 = tf.layers.max_pooling2d(conv4_3, pool_size=2, strides=2, padding='SAME')

		conv5_1 = self.conv2d(max_pool4, num_filters=512, name='conv5_1', trainable=trainable)
		conv5_2 = self.conv2d(conv5_1, num_filters=512, name='conv5_2', trainable=trainable)
		conv5_3 = self.conv2d(conv5_2, num_filters=512, name='conv5_3', trainable=trainable)
		max_pool5 = tf.layers.max_pooling2d(conv5_3, pool_size=2, strides=2, padding='SAME')

		fc6 = tf.contrib.layers.flatten(max_pool5)
		fc6 = self.fc(fc6, size=4096, name='fc6', trainable=trainable)
		fc7 = self.fc(fc6, size=4096, name='NATIN', trainable=trainable)
		self.out = tf.layers.dense(fc7, units=self.num_classes, name='fc8')

	def conv2d(self, layer, num_filters, name, ksize=3, trainable=True):
		return tf.layers.conv2d(layer, filters=num_filters, kernel_size=ksize, 
									activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
									kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
									padding='SAME', trainable=trainable, use_bias=True, name=name)

	def fc(self, layer, size, name, trainable):
		return tf.layers.dense(layer, size, activation=tf.nn.relu, 
								trainable=trainable, kernel_initializer=tf.contrib.layers.xavier_initializer(),
								use_bias=True, name=name)

	def train(self, batch_size, epochs, batch_generator_train, batch_generator_val):
		print("Starting Training...")
		epoch_error, val_error = [], []
		for epoch in range(epochs):
			start_time = time.time()
			train_loss = 0
			val_loss = 0
			batch_num = 0
			for img, lab in batch_generator_train(batch_size):

				batch_time = time.time()
				_, los, err = self.sess.run([self.train_op, self.loss, self.error], feed_dict = {self.x: img,
																								self.y: lab})

				if batch_num % 50 == 0:
					## An error of 2.302585 refelcts random guessing i.e. log(10)=2.302585
					print("Batch {0} Loss {1} Time {2:0f} seconds ".format(batch_num, err, (time.time() - batch_time)*10))

				train_loss += err
				batch_num += 1

			for im, la in batch_generator_val(batch_size):

				ve = self.sess.run([self.error], feed_dict = {self.x: im,
																self.y: la})

				val_loss += ve[0]

			epoch_error.append((train_loss/batch_num))
			val_error.append((val_loss/batch_num))

			print("Epoch {0} Train Loss: {1:1f} Validation Loss: {2} Time: {3}".format(epoch+1, train_loss/batch_num, val_loss/batch_num, time.time()-start_time))

			# Save after every 10 epochs
			if epoch % self.save_epoch == 0:
				self.save_model()

		return epoch_error, val_error


	def predict_single(self, image):
		image = preprocess(image) # preprocess to fit trained network
		image = np.reshape(image, [1, image.shape[0], image.shape[1], 1]) # reshape to [1, 120, 140, 1]
		pred = tf.nn.softmax(self.out)
		pred_lab = tf.argmax(pred, 1)
		prediction, pred_label = self.sess.run([pred, pred_lab], feed_dict={self.x: image})
		return prediction, pred_label

	def save_model(self):
		saver = tf.train.Saver()
		saver.save(self.sess, self.save_path + 'model_' + self.model + '.ckpt')

	def load_model(self):
		saver = tf.train.Saver()
		saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))



if __name__ == '__main__':
	main()
