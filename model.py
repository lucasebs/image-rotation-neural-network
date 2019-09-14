from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

class Model(object):
	def build(classes):
		model = Sequential()
		model.add(Dense(1024, input_shape=(12288,), activation="sigmoid"))
		model.add(Dense(512, activation="sigmoid"))
		model.add(Dense(classes, activation="softmax"))

		INIT_LR = 0.01

		print("[INFO] training network...")
		optimizer = SGD(lr=INIT_LR)
		model.compile(loss="categorical_crossentropy", optimizer=optimizer,
			metrics=["accuracy"])

		return model
