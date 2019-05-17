from __future__ import absolute_import, division, print_function

from matplotlib import pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import os

from tensorflow.keras import layers
from tensorflow.keras import optimizers

import pandas as pd
import numpy as np

tf.VERSION

class CollectBatchStats(tf.keras.callbacks.Callback):
	def __init__(self):
		self.batch_losses = []
		self.batch_acc = []
		self.epoch_val_loss = []
		self.epoch_val_acc = []
	
	def on_batch_end(self, batch, logs=None):
		self.batch_losses.append(logs['loss'])
		self.batch_acc.append(logs['acc'])
	def on_epoch_end(self, epoch, logs=None):
		self.epoch_val_loss.append(logs['val_loss'])
		self.epoch_val_acc.append(logs['val_acc'])

runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
activation = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
size = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
lossF = ['categorical_crossentropy', 'mean_squared_error', 'categorical_crossentropy', 'mean_squared_error', 'categorical_crossentropy', 'mean_squared_error',
	'categorical_crossentropy', 'mean_squared_error', 'categorical_crossentropy', 'mean_squared_error', 'categorical_crossentropy', 'mean_squared_error']
saveName = ['Tanh', 'MSETanh', 'RELU', 'MSERELU', 'LeakyRELU', 'MSELeakyRELU',
	'TanhBig', 'MSETanhBig', 'RELUBig', 'MSERELUBig', 'LeakyRELUBig', 'MSELeakyRELUBig']

for id in runs:
	image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
	train_data = image_generator.flow_from_directory("flower_photos", target_size = (32,32) if size[id] == 0 else (64, 64))
	validation_data = image_generator.flow_from_directory("flower_photos_test", target_size = (32,32) if size[id] == 0 else (64, 64))
	
	for image_batch,label_batch in train_data:
		print("Image batch shape: ", image_batch.shape)
		print("Labe batch shape: ", label_batch.shape)
		break
	
	model = None
	if (activation[id] == 0):
		# CNN 2
		model = tf.keras.Sequential([
		  layers.Conv2D(32, [3, 3], (1, 1), 'valid', input_shape=(image_batch.shape[1], image_batch.shape[2], image_batch.shape[3])),
		  layers.Activation('tanh'),
		  # layers.LeakyReLU(alpha=0.3),
		  layers.MaxPooling2D(pool_size=(2 , 2)),

		  layers.Conv2D(64, [3, 3], (1, 1), 'valid'),
		  layers.Activation('tanh'),
		  # layers.LeakyReLU(alpha=0.3),
		  layers.MaxPooling2D(pool_size=(2 , 2)),

		  layers.Conv2D(128, [3, 3], (1, 1), 'valid'),
		  layers.Activation('tanh'),
		  # layers.LeakyReLU(alpha=0.3),
		  layers.Flatten(),

		  layers.Dense(train_data.num_classes, activation='softmax')
		])
	if (activation[id] == 1):
		# CNN 2
		model = tf.keras.Sequential([
		  layers.Conv2D(32, [3, 3], (1, 1), 'valid', input_shape=(image_batch.shape[1], image_batch.shape[2], image_batch.shape[3])),
		  layers.Activation('relu'),
		  # layers.LeakyReLU(alpha=0.3),
		  layers.MaxPooling2D(pool_size=(2 , 2)),

		  layers.Conv2D(64, [3, 3], (1, 1), 'valid'),
		  layers.Activation('relu'),
		  # layers.LeakyReLU(alpha=0.3),
		  layers.MaxPooling2D(pool_size=(2 , 2)),

		  layers.Conv2D(128, [3, 3], (1, 1), 'valid'),
		  layers.Activation('relu'),
		  # layers.LeakyReLU(alpha=0.3),
		  layers.Flatten(),

		  layers.Dense(train_data.num_classes, activation='softmax')
		])
	if (activation[id] == 2):
		# CNN 2
		model = tf.keras.Sequential([
		  layers.Conv2D(32, [3, 3], (1, 1), 'valid', input_shape=(image_batch.shape[1], image_batch.shape[2], image_batch.shape[3])),
		  # layers.Activation('relu'),
		  layers.LeakyReLU(alpha=0.3),
		  layers.MaxPooling2D(pool_size=(2 , 2)),

		  layers.Conv2D(64, [3, 3], (1, 1), 'valid'),
		  # layers.Activation('relu'),
		  layers.LeakyReLU(alpha=0.3),
		  layers.MaxPooling2D(pool_size=(2 , 2)),

		  layers.Conv2D(128, [3, 3], (1, 1), 'valid'),
		  # layers.Activation('relu'),
		  layers.LeakyReLU(alpha=0.3),
		  layers.Flatten(),

		  layers.Dense(train_data.num_classes, activation='softmax')
		])
	model.compile(
		optimizer=tf.train.AdamOptimizer(), 
		loss=lossF[id],
		metrics=['accuracy'])
	
	batch_stats = CollectBatchStats()
	model.fit_generator(train_data, epochs=16, 
		callbacks = [batch_stats],
		validation_data = validation_data)
	
	results = pd.DataFrame({"losses":batch_stats.batch_losses, "accuracy":batch_stats.batch_acc})
	results.to_csv("saved_results/cnn"+saveName[id]+"Data.csv", ',')
	results = pd.DataFrame({"losses":batch_stats.epoch_val_loss, "accuracy":batch_stats.epoch_val_acc})
	results.to_csv("saved_results/cnn"+saveName[id]+"ValData.csv", ',')
