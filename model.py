import keras

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD

class Model(object):
	# def build(classes):
	# 	model = Sequential()
	# 	model.add(Dense(1024, input_shape=(12288,), activation="sigmoid"))
	# 	model.add(Dense(512, activation="sigmoid"))
	# 	model.add(Dense(classes, activation="softmax"))

	# 	INIT_LR = 0.01

	# 	print("[INFO] training network...")
	#	optimizer = SGD(lr=INIT_LR)
	# 	model.compile(loss="categorical_crossentropy", optimizer=optimizer,
	# 		metrics=["accuracy"])

	# 	return model

	def build(classes):
		model = Sequential()
		model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
		model.add(Activation('relu'))
		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(64, (3, 3), padding='same'))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(classes))
		model.add(Activation('softmax'))

		# initiate RMSprop optimizer
		opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

		# Let's train the model using RMSprop
		model.compile(loss='categorical_crossentropy',
		              optimizer=opt,
		              metrics=['accuracy'])

		return model