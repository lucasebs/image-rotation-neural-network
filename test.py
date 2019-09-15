import gc
import os
import cv2
import csv
import pickle
import argparse
import numpy as np
import pandas as pd

from zipfile import ZipFile
from keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to images directory")
ap.add_argument("-m", "--model", required=True,
	help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label")
ap.add_argument("-f", "--csv", required=True,
	help="path to csv file with preds")
args = vars(ap.parse_args())


path = args["image"]
cont = 0
images = []

df = pd.DataFrame(columns=['fn','label'])

model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

z = ZipFile("corrected_orientation.zip","w")

for r, d, f in os.walk(path):
	for fn in f:
		print(cont)
		image = cv2.imread(path+fn)
		(height, width) = image.shape[:2]
		# output = image.copy()
		output_rotated = image.copy()
		image = cv2.resize(image, (64, 64))
		 
		image = image.astype("float") / 255.0

		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

		preds = model.predict(image)

		i = preds.argmax(axis=1)[0]
		label = lb.classes_[i]

		center = (width / 2, height / 2)
		scale = 1

		# scale_text = 0.4
		# pos_text = (3,13)
		# color_text = (255,255,255)
		# color_text = (0,0,255)
		if label == 'rotated_left':
			M = cv2.getRotationMatrix2D(center, -90, scale)
			output_rotated = cv2.warpAffine(output_rotated, M, (height, width))
			# text = "{}: {:.1f}%".format("L", preds[0][i] * 100)
			# cv2.putText(output, text, pos_text, cv2.FONT_HERSHEY_SIMPLEX, scale_text,
			# 	color_text, 1)
		elif label == 'rotated_right':
			M = cv2.getRotationMatrix2D(center, 90, scale)
			output_rotated = cv2.warpAffine(output_rotated, M, (height, width))
			# text = "{}: {:.1f}%".format("R", preds[0][i] * 100)
			# cv2.putText(output, text, pos_text, cv2.FONT_HERSHEY_SIMPLEX, scale_text,
			# 	color_text, 1)
		elif label == 'upside_down':
			M = cv2.getRotationMatrix2D(center, 180, scale)
			output_rotated = cv2.warpAffine(output_rotated, M, (height, width))
			# text = "{}: {:.1f}%".format("D", preds[0][i] * 100)
			# cv2.putText(output, text, pos_text, cv2.FONT_HERSHEY_SIMPLEX, scale_text,
			# 	color_text, 1)
		# else:
		# 	text = "{}: {:.1f}%".format("U", preds[0][i] * 100)
		# 	cv2.putText(output, text, pos_text, cv2.FONT_HERSHEY_SIMPLEX, scale_text,
		# 		color_text, 1)


		# cv2.imwrite("results/"+fn.split(".")[0]+".png", output)
		image_path = "results_rotated/"+fn.split(".")[0]+"_rotated.png"
		cv2.imwrite(image_path, output_rotated)
		images.append(output_rotated)

		z.write(image_path)

		df.loc[cont] = [fn,label]

		cv2.destroyAllWindows()

		cont += 1
		gc.collect()

z.close()

df.to_csv(args["csv"],index=False, sep=',')
np.save('images_array.npy',np.array(images))