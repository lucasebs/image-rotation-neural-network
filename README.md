# Image Rotation Identification with Neural Network - Based in a simple use of CIFAR10 model. 

The project read an truth file with the images rotation information, open all images and store with the rotation information as labels, build an exact model of [Keras CIFAR10](https://keras.io/examples/cifar10_cnn/) and train the model.

### train.py
- train.py --truth <"path to csv file with truth data"> --model <"path to Keras model"> --label-bin <"path to label">

Before trained the model is applied in a test, reading all the files of a directory image and create a csv file with the predictions. All the images is corrected an storade in an folder and also in a zip file.

### test.py
- test.py --image <"path to images directory"> --model <"path to trained Keras model"> --label-bin <"path to label"> --csv <"path to csv file with preds">

## Training Loss and Accuracy - Simple Neural Network based in CIFAR 10 model.

![Training Loss and Accuracy](https://raw.githubusercontent.com/lucasebs/image-rotation-neural-network/master/fig.png)

