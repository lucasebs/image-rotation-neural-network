# image-rotation-neural-network
Image Rotation Identification with Neural Network

## Use:
### train.py
- train.py --truth <path to csv file with truth data> --model <path to Keras model> --label-bin <path to label>

### test.py
- test.py --image <path to images directory> --model <path to trained Keras model> --label-bin <path to label> --csv <"path to csv file with preds>
