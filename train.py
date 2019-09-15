import csv
import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from model import Model
# from model import model

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--truth", required=True,
	help="path to csv file with truth data")
ap.add_argument("-m", "--model", required=True,
	help="path to Keras model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label")
args = vars(ap.parse_args())


EPOCHS = 50
BS = 64

def load_data_and_labels(fn):
	print("[INFO] loading images...")
	data = []
	labels = []
	cont = 0
	for row in csv.DictReader(open(fn)):
		print(cont)
		img = cv2.imread("train/"+row['fn'])
		data.append(img)
		labels.append(row['label'])
		cont+= 1

	data = np.array(data, dtype="float" ) / 255.0
	labels = np.array(labels)
	return data, labels

data, labels = load_data_and_labels(args['truth'])

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

model = Model.build(len(lb.classes_))

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy - NN unsing CIFAR10 model")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('fig.png')

model.save(args['model'])
f = open(args['label_bin'], "wb")
f.write(pickle.dumps(lb))
f.close()