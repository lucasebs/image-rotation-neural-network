# import the necessary packages
from keras.models import load_model
import argparse
import pickle
import csv
import cv2

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image we are going to classify")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained Keras model")
# ap.add_argument("-l", "--label-bin", required=True,
# 	help="path to label binarizer")
# ap.add_argument("-w", "--width", type=int, default=28,
# 	help="target spatial dimension width")
# ap.add_argument("-e", "--height", type=int, default=28,
# 	help="target spatial dimension height")
# ap.add_argument("-f", "--flatten", type=int, default=-1,
# 	help="whether or not we should flatten the image")
# args = vars(ap.parse_args())


# load the input image and resize it to the target spatial dimensions
fn = "90-10241990_1984-11-28_2007.jpg"
image = cv2.imread("test/"+fn)
(height, width) = image.shape[:2]
output = image.copy()
# image = cv2.resize(image, (args["width"], args["height"]))
 
# scale the pixel values to [0, 1]
image = image.astype("float") / 255.0

# check to see if we should flatten the image and add a batch
# dimension
# if args["flatten"] > 0:
# 	image = image.flatten()
# 	image = image.reshape((1, image.shape[0]))

image = image.flatten()
image = image.reshape((1, image.shape[0]))
 
# otherwise, we must be working with a CNN -- don't flatten the
# image, simply add the batch dimension
# else:
# 	image = image.reshape((1, image.shape[0], image.shape[1],
# 		image.shape[2]))

print("[INFO] loading network and label binarizer...")
model = load_model("model.model")
lb = pickle.loads(open("labels.pickle", "rb").read())
 
# make a prediction on the image
preds = model.predict(image)
 
# find the class label index with the largest corresponding
# probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
# cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
# 	(0, 0, 255), 2)

center = (width / 2, height / 2)
scale = 1

output_rotated = output

if label == 'rotated_left':
	M = cv2.getRotationMatrix2D(center, -90, scale)
	output_rotated = cv2.warpAffine(output, M, (height, width))
elif label == 'rotated_right':
	M = cv2.getRotationMatrix2D(center, 90, scale)
	output_rotated = cv2.warpAffine(output, M, (height, width))
elif label == 'upside_down':
	M = cv2.getRotationMatrix2D(center, 180, scale)
	output_rotated = cv2.warpAffine(output, M, (height, width))

print(text)
# show the output image
# cv2.imshow("Image", output)
# cv2.imwrite("image.png", output)
cv2.imwrite("image_corrected.png", output_rotated)
# cv2.waitKey(0)


with open('test.preds.csv', mode='w') as csv_file:
	fieldnames = ['fn','label']
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
	writer.writeheader()
	writer.writerow({'fn':fn, 'label':label})
