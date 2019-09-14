import csv
import cv2
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


def load_data_and_labels(fn):
	#initialize data and labels
	print("[INFO] loading images...")
	data = []
	labels = []
	cont = 0
	for row in csv.DictReader(open(fn)):
		img = cv2.imread("train/"+row['fn'])
		data.append(img)
		labels.append(row['label'])
		if cont == 100:
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

print(trainY)
print(testY)