import csv
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from model import Model
# from model import model

EPOCHS = 75
BS = 64

def load_data_and_labels(fn):
	#initialize data and labels
	print("[INFO] loading images...")
	data = []
	labels = []
	cont = 0
	for row in csv.DictReader(open(fn)):
		img = cv2.imread("train/"+row['fn'])
		# img = cv2.imread("train/"+row['fn']).flatten()
		data.append(img)
		labels.append(row['label'])
		if cont == 3000:
			break
		cont+= 1

	data = np.array(data, dtype="float" ) / 255.0
	labels = np.array(labels)
	return data, labels

data, labels = load_data_and_labels('train.truth.csv')

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

model = Model.build(len(lb.classes_))
# print(trainY)
# print(testY)

# H = model.fit(trainX, trainY, validation_data=(testX, testY),
# 	epochs=EPOCHS, batch_size=64)

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('fig.png')

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save("model.model")
f = open("labels.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()